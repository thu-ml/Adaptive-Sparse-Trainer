import torch.nn as nn
from dataclasses import dataclass
from torch.nn import Parameter
from torch.nn import functional as F
import torch
import re
import copy
import wandb

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
    


@dataclass
class SparseLinearConfig:
    change_mask: bool = True
    mask_metric: str = "magnitude"  
    SLoRB: bool = True
    SLoRB_k: int = 64
    mask_type: str = "structured"
    SLoRB_init_type: str = "mean"
    trainable_projection: bool = False
    gradient_checkpointing: bool = False
    mode: str = "sparse_forward"




class SparseLinear(nn.Linear):
    """
    Note that for c_attn weight the key query value matrix share the same sclaer_row however since we are using N:M sparsity 
    there is no need to use different scaler_row for the key and value matrix.
    """
    def __init__(self, in_features: int, out_features: int, sparselinear_config=None, bias: bool = True, **kwargs):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        assert sparselinear_config is not None
        self.mask_type = sparselinear_config.mask_type
        self.mask_metric = sparselinear_config.mask_metric # choose from ["wanda", "magnitude"]
        self.nsamples = 0
        self.mode = sparselinear_config.mode
        self.change_mask = sparselinear_config.change_mask
        self.SLoRB = sparselinear_config.SLoRB


        if self.SLoRB:
            self.SLoRB_k = sparselinear_config.SLoRB_k
            self.SLoRB_init_type = sparselinear_config.SLoRB_init_type
            self.trainable_projection = sparselinear_config.trainable_projection


            

    def initialize(self):
        out_features, in_features = self.weight.shape

        self.weight.mask = torch.ones_like(self.weight).bool()
        self.weight.init_mask = torch.ones_like(self.weight).bool()
        self.weight.param_count = in_features * out_features
        self.weight.flipped_mask = 0
        self.weight.init_flipped_mask = 0

        new_mask = self.calculate_mask()
        
        self.weight.mask = new_mask
        self.weight.init_mask = new_mask


    def update_mask(self):
        if self.change_mask:
            new_mask = self.calculate_mask()
            self.weight.init_flipped_mask = torch.sum(new_mask != self.weight.init_mask).item()
            self.weight.flipped_mask = torch.sum(new_mask != self.weight.mask).item()
            self.weight.mask = new_mask.int()
        else:
            self.weight.init_flipped_mask = 0
            self.weight.flipped_mask = 0

    def init_SLoRB(self):
        N , d = self.weight.shape
        # init x_projtion matrix
        rows = d // self.SLoRB_k
        cols = d
        x_proj = torch.zeros(rows, cols)
        indices = torch.arange(rows) * self.SLoRB_k
        x_proj[torch.arange(rows)[:, None], indices[:, None] + torch.arange(self.SLoRB_k)] = 1
        x_proj.float()
        self.x_proj = nn.Parameter(x_proj, requires_grad=self.trainable_projection)
        shape = (N, d // self.SLoRB_k)
        self.SLoRB_Weight = nn.Parameter(torch.zeros(shape, requires_grad=True))

        if self.SLoRB_init_type == "xavier":
            nn.init.xavier_uniform_(self.x_proj)
        else:
            pruned_weight = self.weight.data.clone().detach() * (1-self.weight.mask)
            pruned_weight = pruned_weight.view(pruned_weight.shape[0], pruned_weight.shape[1] // self.SLoRB_k, self.SLoRB_k)
            if self.SLoRB_init_type == "mean":
                pruned_weight_mean = pruned_weight.mean(dim=2)
            elif self.SLoRB_init_type == "sum":
                pruned_weight_mean = pruned_weight.sum(dim=2)
            else:
                raise ValueError("Invalid SLoRB init type")
            self.SLoRB_Weight.data = pruned_weight_mean



    def sync_weight(self):
        self.weight.mask = self.weight.mask.to(self.weight.device)
        self.weight.init_mask = self.weight.init_mask.to(self.weight.device)

    def forward(self, x):
        if self.mode == "dense_forward":
            model_ouput = F.linear(x, self.weight, self.bias)
        elif self.mode == "sparse_forward":
            masked_weight =  STE.apply(self.weight, self.weight.mask)
            model_ouput = F.linear(x, masked_weight, self.bias)
        else:
            raise ValueError("Invalid mode")
        if self.SLoRB and hasattr(self, "x_proj"):
            lora_output = F.linear(x , self.x_proj, bias=None)
            lora_output = F.linear(lora_output, self.SLoRB_Weight, bias=None)
            model_ouput = model_ouput + lora_output


        return model_ouput
    
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
    def calculate_mask(self):
        mask_type = self.mask_type
        data = self.weight.data.clone().detach()

        if self.mask_metric == "wanda":
            W_metric = torch.abs(data) * torch.sqrt(self.scaler_row.reshape((1,-1)))
        elif self.mask_metric == "magnitude":
            W_metric = torch.abs(data)
        else:
            raise ValueError("Invalid mask metric")


        new_mask = (torch.zeros_like(W_metric) == 1)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)

        if mask_type == "unstructured":
            sparsity_ratio = 0.5
            # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            new_mask.scatter_(1, indices, True)
            new_mask = (~new_mask).int()
        elif mask_type == "structured":
            self.N = 2
            self.M = 4
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % self.M == 0:
                    tmp = W_metric[:,ii:(ii+self.M)].float()
                    new_mask.scatter_(1,ii+torch.topk(tmp, self.M-self.N ,dim=1, largest=False)[1], True)
            new_mask = (~new_mask).int()
        else:
            raise ValueError("Invalid mask type")
        return new_mask



    

class Distill_Model(torch.nn.Module):
    def __init__(self, model, teacher=None, output_hidden_state=False):
        super(Distill_Model, self).__init__()
        self.student = model
        self.teacher = teacher
        self.student.config.output_hidden_state = output_hidden_state
        self.teacher.config.output_hidden_state = output_hidden_state
        self.output_hidden_state = output_hidden_state
        self.teacher.eval()
    
    def forward(self, idx, targets=None):
        if self.output_hidden_state:
            student_logits, task_loss, student_hidden_states = self.student(idx, targets)

            with torch.no_grad():
                teacher_logits, _, teacher_hidden_states = self.teacher(idx, targets)
            if student_hidden_states is not None and teacher_hidden_states is not None:
                layerwise_loss = self.layerwise_loss(student_hidden_states, teacher_hidden_states)
            else:
                layerwise_loss = 0.0
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            return student_logits, task_loss, layerwise_loss, kl_loss
        else:
            student_logits, task_loss = self.student(idx, targets)
            with torch.no_grad():
                teacher_logits, _ = self.teacher(idx, targets)
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            return student_logits, task_loss, None, kl_loss
    
    def kl_loss(self, student_logits, teacher_logits, temperature=2):
        num_tokens = student_logits.numel() / student_logits.size(-1)
        kl_loss = F.kl_div(
            input = F.log_softmax(student_logits / temperature, dim=-1),
            target = F.log_softmax(teacher_logits / temperature, dim=-1),
            log_target=True,
            reduction="sum",
        ) * (temperature**2)/ num_tokens
        return kl_loss

    def layerwise_loss(self, student_hidden_states, teacher_hidden_states):
        length = len(student_hidden_states)
        layerwise_loss = 0.0
        for i in range(length):
            layerwise_loss += (student_hidden_states[i] - teacher_hidden_states[i]).pow(2).mean() / (teacher_hidden_states[i].pow(2).mean() + torch.finfo(torch.bfloat16).eps)
        return layerwise_loss