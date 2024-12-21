import torch 
from sparse_modeling import SparseLinear
from model import GPT, GPTConfig
from datasets import load_dataset
import tiktoken
import torch.nn as nn
import pickle


@torch.no_grad()
def get_raw_model(model):
    if hasattr(model, "module"):
        if hasattr(model.module, "student"):
            raw_model = model.module.student
        else:
            raw_model = model.module
    else:
        if hasattr(model, "student"):
            raw_model = model.student
        else:
            raw_model = model
    return raw_model


@torch.no_grad()
def calculate_model_mask(model):
    model = get_raw_model(model)            
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.update_mask()


@torch.no_grad()
def calculate_flip_rate(model):
    model = get_raw_model(model) 


    flipped = 0
    init_flipped = 0
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            flipped += module.weight.flipped_mask
            init_flipped += module.weight.init_flipped_mask
            total += module.weight.param_count
    return flipped ,flipped / total, init_flipped, init_flipped / total
@torch.no_grad()
def init_mask(model, mask_type):
    model = get_raw_model(model) 
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.init_mask(mask_type)
@torch.no_grad()
def set_model_mode(model, mode):
    model = get_raw_model(model) 
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.mode = mode


@torch.no_grad()
def initialize_model(model):
    model = get_raw_model(model) 
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.initialize()
            if module.SLoRB:
                module.init_SLoRB()

def sync_weight(model, device):
    model = model.to(device)
    model = get_raw_model(model) 
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.sync_weight()


@torch.no_grad()
def eval_ppl(model, bs=2, device="cuda:0", block_size=1024):
    testdata = load_dataset('/root/autodl-tmp/nanoGPT/data/wikitext/wikitext-2-raw-v1', split='test',cache_dir=None)
    tokenizer = tiktoken.get_encoding("gpt2")
    testenc = tokenizer.encode_ordinary("\n\n".join(testdata['text']))
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/opt-125m', use_fast=False, trust_remote_code=True)
    # testenc = tokenizer.encode("\n\n".join(testdata['text']))
    model.eval()
    # transfrom list to tensor
    testenc = torch.tensor(testenc, dtype=torch.long, device=device).unsqueeze(0)
    # Calculate number of samples
    nsamples = testenc.numel() // block_size

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * block_size):(j * block_size)].to(device)
        inputs = inputs.reshape(j-i, block_size)

        # Forward pass through the model
        lm_logits = model(inputs)[0]

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * block_size * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * block_size))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    model.train()
    return ppl.item()



def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res 


def prepare_calibration_input(model, dataloader, device, nsamples):

    layers = model.transformer.h
    model = model.to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.config.block_size, model.config.n_embd), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    inps, outs = inps.to(device), outs.to(device)
    return inps, outs

def add_calibration(model, nsamples=128, device="cuda"):

    model = get_raw_model(model)
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            in_features = module.weight.shape[1]
            module.scaler_row = nn.Parameter(torch.ones(in_features, device=device))

    with open('/root/autodl-tmp/nanoGPT/data/c4_dataset/calibration_dataset.pkl', 'rb') as f:
        dataloader = pickle.load(f)

    inps, outs = prepare_calibration_input(model, dataloader, device, nsamples)
    layers = model.transformer.h
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=[SparseLinear])

        def add_batch(name):
            def tmp(_, inp, out):
                subset[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        inps, outs = outs, inps
    torch.cuda.empty_cache()

if __name__ == "__main__":
    n_layer = 12
    n_head = 12
    n_embd = 768
    block_size = 1024
    bias = True
    dropout = 0.1

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    model_args['vocab_size'] =  50304
    # gptconf = GPTConfig(**model_args)
    # model = GPT(gptconf)

    init_from = "gpt2"

    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size


    device = torch.device("cuda:0")
    model = model.to(device)
    print(eval_ppl(model, bs=1, device=device, block_size=1024))

    add_calibration(model, nsamples=128, device="cuda:0")
    for n,p in model.named_parameters():
        if hasattr(p, 'mask'):
            p.mode = "2:4"
            
    print(eval_ppl(model, bs=1, device=device, block_size=1024))
            