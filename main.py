"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sparse_modeling import  Distill_Model, SparseLinearConfig
from utils import calculate_model_mask, calculate_flip_rate, set_model_mode, initialize_model, add_calibration, eval_ppl, sync_weight
from model import GPTConfig, GPT
import os
import argparse


parser = argparse.ArgumentParser(description='Arguments for training GPT')
# Distillation arguments
parser.add_argument('--distill_model', type=bool, default=False)
parser.add_argument('--teacher_model', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
parser.add_argument('--student_model', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
parser.add_argument('--hardness_task', type=float, default=1.0)
parser.add_argument('--hardness_kldiv', type=float, default=1.0)
parser.add_argument('--hardness_squarehead', type=float, default=1.0)
# Training arguments
parser.add_argument('--eval_interval', type=int, default=200)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--eval_iters', type=int, default=20)
parser.add_argument('--output_flip_every', type=int, default=10)
# Hyperparameters
parser.add_argument('--global_batch_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-1)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--srste_decay', type=float, default=6e-5)
# Iteration settings
parser.add_argument('--max_iters', type=int, default=20000)
parser.add_argument('--warmup_iters', type=int, default=1000)
parser.add_argument('--lr_decay_iters', type=int, default=20000)
parser.add_argument('--increase_step', type=int, default=10000)
# Masking arguments
parser.add_argument('--mode', choices=['sparse_forward', 'dense_forward'], default='sparse_forward')
parser.add_argument('--mask_type', choices=['structured', 'unstructured'], default='structured')
parser.add_argument('--mask_metric', choices=['wanda', 'magnitude'], default='magnitude')
parser.add_argument('--change_mask', type=bool, default=False)
# SLoRB arguments
parser.add_argument('--SLoRB_k', type=int, default=64)
parser.add_argument('--SLoRB', type=bool, default=False)
parser.add_argument('--SLoRB_init_type', choices=['mean', 'sum', 'xavier'], default='mean')
parser.add_argument('--trainable_projection', type=bool, default=False)
# Other arguments
parser.add_argument('--gradient_checkpointing', type=bool, default=False)
parser.add_argument('--wandb_logging', type=bool, default=False)
args=parser.parse_args()

print(args)

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
gradient_checkpointing = args.gradient_checkpointing
eval_interval = args.eval_interval
log_interval = args.log_interval
eval_iters = args.eval_iters
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
teacher_init_from = args.teacher_model # 'scratch' or 'resume' or 'gpt2*'
student_init_from = args.student_model # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = args.wandb_logging # disabled by default
wandb_project = f"Github-Repo"
# data
dataset = 'c4_dataset'
batch_size = args.batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
learning_rate = args.learning_rate # max learning rate
max_iters = args.max_iters # total number of training iterations
weight_decay = args.weight_decay # strength of weight decay
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = args.warmup_iters # how many steps to warm up for
lr_decay_iters = args.lr_decay_iters  # should be ~= max_iters per Chinchilla
min_lr = args.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
wandb_run_name = f"gpt2"
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
hardness_task = args.hardness_task
hardness_kldiv = args.hardness_kldiv
hardness_squarehead = args.hardness_squarehead
output_flip_every = args.output_flip_every
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
assert args.global_batch_size % batch_size == 0
gradient_accumulation_steps = int(args.global_batch_size / batch_size) # used to simulate larger batch sizes
# -----------------------------------------------------------------------------

# in order to calculate the layerwise loss the teacher and student model must output hidden states
if hardness_squarehead != 0:
    assert teacher_init_from == student_init_from
# if layerwise loss is well defined, we calculate and log it no matter what
if teacher_init_from == student_init_from:
    output_hidden_state = True
else:
    output_hidden_state = False

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_wiki_ppl = 1e9



# model init



sparselinear_config = SparseLinearConfig(change_mask=args.change_mask, mask_metric=args.mask_metric, mask_type=args.mask_type,
                          SLoRB=args.SLoRB, SLoRB_k=args.SLoRB_k, SLoRB_init_type=args.SLoRB_init_type, 
                          trainable_projection=args.trainable_projection, mode=args.mode)

override_args = dict(dropout=dropout, output_hidden_state=output_hidden_state, gradient_checkpointing=gradient_checkpointing)
if args.distill_model:
    print(f"Initializing teacher model from OpenAI GPT-2 weights: {teacher_init_from}")
    teacher_model = GPT.from_pretrained(teacher_init_from, is_teacher=True, override_args=override_args)

print(f"Initializing model from OpenAI GPT-2 weights: {student_init_from}")
student_model = GPT.from_pretrained(student_init_from, is_teacher=False, override_args=override_args, sparselinear_config=sparselinear_config)

student_model.eval()

# if we are using the WANDA mask metric, we need to intialize scaler row first 
if args.mask_metric == 'wanda':
    with torch.no_grad():
        set_model_mode(student_model, mode='dense_forward')
        add_calibration(student_model,  device=device)
        set_model_mode(student_model, mode=args.mode)

initialize_model(student_model)
student_model.train()
if hasattr(student_model.config, "block_size") and block_size < student_model.config.block_size:
    student_model.crop_block_size(block_size)


if args.distill_model:
    model = Distill_Model(student_model, teacher_model, output_hidden_state=output_hidden_state)
else:
    model = student_model

# set_model_mode(student_model, mode='sparse_forward')
# eval_ppl(student_model, bs=2, device=device, block_size=1024)



# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer

if args.distill_model:
    # optimizer = model.student.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, srste_decay=args.srste_decay)
    optimizer = model.student.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
else:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# make sure mask and weights are on the same device
sync_weight(model, device)


checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# model.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if args.distill_model:  
                    logits, task_loss, layerwise_loss, kl_loss = model(X, Y)
                    if task_loss is None:
                        task_loss = 0
                    if layerwise_loss is None:
                        loss = hardness_task * task_loss + hardness_kldiv * kl_loss
                    else:
                        loss = hardness_task * task_loss + hardness_squarehead * layerwise_loss + hardness_kldiv * kl_loss
                else:
                    logits, loss, hidden_states = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        assert max_iters >= lr_decay_iters
        return min_lr * ( max_iters - it ) / (max_iters - lr_decay_iters)
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# annealing srste for decay    
def get_decay(it):
    if it < args.increase_step:
        decay = args.srste_decay / args.increase_step * it
    else:
        decay = args.srste_decay  
    return decay

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed


temp = None
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    decay = get_decay(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        wiki_ppl = eval_ppl(model, bs=2, device=device, block_size=1024)
        print(f"evaluating: iter_num {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, wiki_ppl {wiki_ppl:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val/loss": losses['val'],
                "lr": lr,
                "wiki_ppl": wiki_ppl,
            })
        if wiki_ppl < best_wiki_ppl or always_save_checkpoint:
            if os.path.exists(os.path.join(out_dir, f"{best_wiki_ppl}_ckpt.pt")):
                os.remove(os.path.join(out_dir, f"{best_wiki_ppl}_ckpt.pt"))
            best_val_loss = losses['val']
            best_wiki_ppl = wiki_ppl

            if args.distill_model:
                save_model = raw_model.student
            else:
                save_model = raw_model
            if iter_num > 0:
                checkpoint = {
                    'model': save_model.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f"{best_wiki_ppl}_ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if args.distill_model:
                logits, task_loss, layerwise_loss, kl_loss = model(X, Y)
                if task_loss is None:
                    task_loss = 0.0
                if layerwise_loss is None:
                    loss = hardness_task * task_loss + hardness_kldiv * kl_loss  
                else:
                    loss = hardness_task * task_loss + hardness_squarehead * layerwise_loss + hardness_kldiv * kl_loss 
            else:
                logits, loss, hidden_states = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer,decay=decay)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % output_flip_every == 0:

        calculate_model_mask(model)

    if iter_num % output_flip_every == 0 and master_process:
        flipped, flipped_ratio, init_flipped, init_flipped_ratio = calculate_flip_rate(model) 
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if args.distill_model:
            task_lossf = task_loss.item()
            if layerwise_loss is not None:
                layerwise_lossf = layerwise_loss.item()
            else:
                layerwise_lossf = 0
            kl_lossf = kl_loss.item()
            print(f"iter_num: {iter_num}, flip_num: {flipped}, flip_ration: {flipped_ratio}, init_flip_num: {init_flipped}, init_flip_ration: {init_flipped_ratio}, loss: {lossf:.4f}, time: {dt*1000:.2f}ms")
            if wandb_log:
                wandb.log({
                        "iter": iter_num,
                        "flip_num": flipped,
                        "init_flip_num": init_flipped,
                        "flip_ration": flipped_ratio,
                        "init_flip_ration": init_flipped_ratio,
                        "train/loss": lossf,
                        "train/task_loss": task_lossf,
                        "train/layerwise_loss": layerwise_lossf,
                        "train/kl_loss": kl_lossf,
                        "lr": lr,
                        "time": dt*1000,
                        "srste_decay": decay,
                })
        else:
            print(f"iter_num: {iter_num}, flip_num: {flipped}, flip_ration: {flipped_ratio}, init_flip_num: {init_flipped}, init_flip_ration: {init_flipped_ratio}, loss: {lossf:.4f}, time: {dt*1000:.2f}ms")
            if wandb_log:
                wandb.log({
                        "iter": iter_num,
                        "flip_num": flipped,
                        "init_flip_num": init_flipped,
                        "flip_ration": flipped_ratio,
                        "init_flip_ration": init_flipped_ratio,
                        "train/loss": lossf,
                        "lr": lr,
                        "time": dt*1000,
                        "srste_decay": decay,
                    })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        print(args)
        break

if ddp:
    destroy_process_group()
