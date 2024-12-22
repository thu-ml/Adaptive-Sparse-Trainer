from datasets import load_dataset
import tiktoken
import torch.nn as nn
import torch
import pickle
from llmshearing.sparse.sparse_modeling import SparseLinear
from transformers import AutoTokenizer, OPTForCausalLM
from llmshearing.sparse.adamw import AdamW

@torch.no_grad()
def eval_ppl(model, bs=2, device="cuda:0", block_size=1024):
    testdata = load_dataset('/data/home/huangweiyu/LLM-Shearing/wikitext', split='test',cache_dir=None)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/data/home/huangweiyu/LLM-Shearing/llmshearing/hf_tokenizer', use_fast=False, trust_remote_code=True)
    testenc = tokenizer.encode("\n\n".join(testdata['text']))
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
        inputs_dic = {'input_ids': inputs, 'labels': inputs}
        # Forward pass through the model
        with torch.no_grad():
            output = model(inputs_dic)
            if isinstance(output, dict):
                lm_logits = output['logits']
            else:
                lm_logits = output[0]

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


def get_nested_attr(obj, attr_path):
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj

def set_nested_attr(obj, attr_path, value):
    attributes = attr_path.split('.')
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)

def get_linear_list(model):
    linear_list = []
    for name, module in model.named_modules():
        unwanted_suffix = "output"
        if isinstance(module, nn.Linear) and not name.endswith(unwanted_suffix):
            linear_list.append(name)
    return linear_list

def get_sparse_model(model, linear_config):
    linear_list = get_linear_list(model)
    for name in linear_list:
        module = get_nested_attr(model, name)
        if  module.bias is None:
            s_linear = SparseLinear(module.in_features, module.out_features, bias=False, 
                                    training_config=linear_config)
            s_linear.weight.data = module.weight.data
        else:
            s_linear = SparseLinear(module.in_features, module.out_features, bias=True, 
                                    training_config=linear_config)
            s_linear.weight.data = module.weight.data
            s_linear.bias.data = module.bias.data
        set_nested_attr(model, name, s_linear)
    torch.cuda.empty_cache()
    return model

def configure_optimizers(model, weight_decay, learning_rate, betas, srste_decay):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    decay_params_names = [n for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params_names = [n for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'name': decay_params_names},
        {'params': nodecay_params, 'weight_decay': 0.0, 'name': nodecay_params_names}

    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, srste_decay=srste_decay, model=model)
    return optimizer