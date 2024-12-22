import torch.nn as nn
# import Parameter
from torch.nn import Parameter
from torch.nn import functional as F
import torch
import re
import copy
import wandb
from llmshearing.sparse.legacy import sparse24_triton
class STE(torch.autograd.Function):
    """ Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, mask):
        
        ctx.save_for_backward(weight)
        ctx.mask = mask
        return weight*mask

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output , None, None
    
class SRSTE(torch.autograd.Function):
    """ Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, mask, decay):
        
        ctx.save_for_backward(weight)
        ctx.mask = mask
        ctx.decay = decay
        return weight*mask

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (ctx.mask) * weight, None, None
    

class SparseLinearConfig:
    def __init__(self,  SLoRA=False, SLoRA_k=64, train_mask = True, calculate_mask_every_forward=10,
                  mask_metric="magnitude", mask_type="calculate_2:4_mask",SLoRA_init_type="mean",trainable_projection=False):
        self.SLoRA = SLoRA
        self.SLoRA_k = SLoRA_k
        self.mask_metric=mask_metric
        self.mask_type = mask_type
        self.SLoRA_init_type = SLoRA_init_type
        self.trainable_projection = trainable_projection
        self.train_mask = train_mask
        self.calculate_mask_every_forward = calculate_mask_every_forward

class SparseLinear(nn.Linear):
    """
    Note that for c_attn weight the key query value matrix share the same sclaer_row however since we are using N:M sparsity 
    there is no need to use different scaler_row for the key and value matrix.
    """
    def __init__(self, in_features: int, out_features: int, training_config=None, bias: bool = True, **kwargs):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        assert training_config is not None
        self.register_buffer('mask', torch.ones(out_features, in_features, dtype=torch.bool))
        self.register_buffer('inital_mask', torch.ones(out_features, in_features, dtype=torch.bool))
        self.param_count = in_features * out_features
        self.flipped_mask = 0
        self.init_flipped_mask = 0
        self.mask_metric = training_config.mask_metric # choose from ["wanda", "magnitude"]
        self.mask_type = training_config.mask_type
        self.nsamples = 0
        self.mode = None
        self.SLoRA = training_config.SLoRA
        self.SLoRA_k = training_config.SLoRA_k
        self.SLoRA_init_type = training_config.SLoRA_init_type
        self.trainable_projection = training_config.trainable_projection
        self.scaler_row = torch.zeros(in_features, dtype=torch.float32)
        self.forward_count = 0
        self.calculate_mask_every_forward=training_config.calculate_mask_every_forward
        self.train_mask = training_config.train_mask
        if self.SLoRA:
            N , d = self.weight.shape
            self.x_proj = nn.Parameter(torch.zeros(d // self.SLoRA_k, d), requires_grad=self.trainable_projection)
            self.SLoRA_Weight = nn.Parameter(torch.zeros(N, d // self.SLoRA_k), requires_grad=True)       
  


    def forward(self, x):
        if self.mode == "sparse_training" and self.training==True:
            self.forward_count += 1
            if self.forward_count % self.calculate_mask_every_forward == 0:
                # print(f"Calculating mask on {self.forward_count}")
                data = self.weight.data.clone().detach().to(dtype=torch.float32)     
                _, mask = sparse24_triton(data)
                
                self.flipped_mask = torch.sum(mask ^ self.mask).item()
                self.init_flipped_mask = torch.sum(mask ^ self.inital_mask).item()
                if self.train_mask:
                    self.mask = mask
                del mask
                del data

        if self.mode == "dense_training":
            x = F.linear(x, self.weight, self.bias)
        elif self.mode == "sparse_training":
            masked_weight =  STE.apply(self.weight, self.mask)
            model_ouput = F.linear(x, masked_weight, self.bias)
            # x = F.linear(x, self.weight * self.mask, self.bias)
            if self.SLoRA:
                lora_output = F.linear(x , self.x_proj, bias=None)
                lora_output = F.linear(lora_output, self.SLoRA_Weight, bias=None)
                x = model_ouput + lora_output
            else:
                x = model_ouput
        else:
            raise ValueError("Invalid mode")
        return x
    

    @torch.no_grad()
    def init_mask(self, mask_type=None):
        if self.weight.device == torch.device("cuda"):
            new_mask = sparse24_triton(self.weight.data)  
        else:
            new_mask = self.current_mask(mask_type)     
        self.inital_mask = new_mask
        self.mask = new_mask
        if self.SLoRA:
            self.init_SLoRA(self.SLoRA_init_type)
        return None




    @torch.no_grad()
    def add_batch(self, inp, out):
        """Set the scaler_row to be the mean of the weight matrix"""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
    
    @torch.no_grad()
    def init_SLoRA(self,SLoRA_init_type):
        if SLoRA_init_type == "xavier":
            nn.init.xavier_uniform_(self.SLoRA_Weight)
            nn.init.xavier_uniform_(self.x_proj)
        else:
            N , d = self.weight.shape
            # init x_projtion matrix
            rows = d // self.SLoRA_k
            cols = d
            x_proj = torch.zeros(rows, cols)
            indices = torch.arange(rows) * self.SLoRA_k
            x_proj[torch.arange(rows)[:, None], indices[:, None] + torch.arange(self.SLoRA_k)] = 1
            x_proj.float()
            self.x_proj = nn.Parameter(x_proj, requires_grad=self.trainable_projection)
            pruned_weight = self.weight.data.clone().detach() * (~self.mask)
            pruned_weight = pruned_weight.view(pruned_weight.shape[0], pruned_weight.shape[1] // self.SLoRA_k, self.SLoRA_k)
            if SLoRA_init_type == "mean":
                pruned_weight_mean = pruned_weight.mean(dim=2)
            elif SLoRA_init_type == "sum":
                pruned_weight_mean = pruned_weight.sum(dim=2)
            elif SLoRA_init_type == "zero":
                pruned_weight_mean = torch.zeros(N, d // self.SLoRA_k)
            else:
                raise ValueError("Invalid SLoRA init type")
            self.SLoRA_Weight.data = pruned_weight_mean


    @torch.no_grad()
    def current_mask(self, mask_type=None, return_W_metric=False):
        if mask_type is not None:
            self.mask_type = mask_type

        data = self.weight.data.clone().detach()
        if self.mask_metric == "wanda":
            W_metric = torch.abs(data) * torch.sqrt(self.scaler_row.reshape((1,-1)))
        elif self.mask_metric == "magnitude":
            W_metric = torch.abs(data)
        else:
            raise ValueError("Invalid mask metric")

        new_mask = torch.zeros_like(W_metric, dtype=torch.bool)
        if self.mask_type == "calculate_dense_mask":
            new_mask = torch.ones_like(self.weight,dtype=torch.bool)
        elif self.mask_type == "calculate_unstructured_mask":
            sparsity_ratio = 0.5
            # unstructured pruning
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            new_mask.scatter_(1, indices, True)
            new_mask = (~new_mask)
        elif matches := re.findall(r"calculate_(\d+):(\d+)_mask", self.mask_type):
            self.N = int(matches[0][0])
            self.M = int(matches[0][1])
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % self.M == 0:
                    tmp = W_metric[:,ii:(ii+self.M)].float()
                    new_mask.scatter_(1,ii+torch.topk(tmp, self.M-self.N ,dim=1, largest=False)[1], True)
            new_mask = (~new_mask)
        else:
            raise ValueError("Invalid mode")
        
        if return_W_metric:
            return new_mask, W_metric
        else:
            return new_mask
    
    @torch.no_grad()
    def calculate_percentage_mask(self, mask_type, change_mask=True, percentage=0.0):

        if self.weight.device == torch.device("cuda"):
            new_mask = sparse24_triton(self.weight.data) 
        else:
            new_mask, W_metric = self.current_mask(mask_type, return_W_metric=True)  
        
        if percentage != 0.0:
            assert re.findall(r"calculate_(\d+):(\d+)_mask", mask_type)
            new_w_metric = (~new_mask) * W_metric
            reshape_w_metric = new_w_metric.view(W_metric.shape[0], W_metric.shape[1] // self.M, self.M)
            sum_tensor = reshape_w_metric.sum(dim=2)
            flat_sum_tensor = sum_tensor.flatten()
            topk = int(percentage * flat_sum_tensor.shape[0])
            _, indices_top_percent = torch.topk(flat_sum_tensor, k=topk)

            value = torch.tensor(1).int().to(new_mask.device)
            rows, cols = torch.div(indices_top_percent, sum_tensor.size(1), rounding_mode='trunc'), indices_top_percent % sum_tensor.size(1)
            rows = rows.repeat(4)
            cols = torch.hstack([4*cols+i for i in range(4)])
            indices = (rows, cols)
            new_mask.index_put_(indices, value.expand_as(rows), accumulate=False)

        self.init_flipped_mask = torch.sum(new_mask ^ self.inital_mask).item()
        self.flipped_mask = torch.sum(new_mask ^ self.mask).item()
        if change_mask:
            self.mask = new_mask
        return None
    
    

@torch.no_grad()
def calculate_model_mask(model, mask_type, change_mask=True, percentage=0.0):
    if hasattr(model, "module"):
        if hasattr(model.module, "student"):
            model = model.module.student
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.calculate_percentage_mask(mask_type, change_mask=change_mask, percentage=percentage)
@torch.no_grad()
def calculate_flip_rate(model):
    if hasattr(model, "module"):
        if hasattr(model.module, "student"):
            model = model.module.student
    flipped = 0
    init_flipped = 0
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            flipped += module.flipped_mask
            init_flipped += module.init_flipped_mask
            total += module.param_count
    return flipped ,flipped / total, init_flipped, init_flipped / total
@torch.no_grad()
def init_mask(model, mask_type):
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.init_mask(mask_type)
@torch.no_grad()
def set_model_mode(model, mode):
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            module.mode = mode


