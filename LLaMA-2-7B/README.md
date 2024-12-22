# Training Script for LLaMA-2-7B 

This codebase is modified from Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning [ArXiv Preprint](https://arxiv.org/abs/2310.06694) | [Blog Post](https://xiamengzhou.github.io/sheared-llama/)  

## Install Requirements
**Step 1**: To get started with this repository, you'll need to follow these installation steps. Before proceeding, make sure you have [Pytorch](https://pytorch.org/get-started/previous-versions/) and [Flash Attention](https://github.com/Dao-AILab/flash-attention)  installed. You can do this via pip using the following commands:
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==1.0.3.post
```
Please note that Flash Attention version 2 is not currently supported and may require manual modifications to the model file. 

**Step 2**: Then install the rest of the required packages:
```
cd llmshearing
pip install -r requirement.txt
```

**Step 3**: Finally, install the `llmshearing` package in editable mode to make it accessible for your development environment:
```
pip install -e .
```


## Dataset and Model Preparation
We use the dataset for the pruning stage, follow the instruction of the orginial code base for instrcutions on how to preapre data and models at this [link](https://github.com/princeton-nlp/LLM-Shearing/blob/main/README.md).


## Configuration

Configurate your run at configs/7b.yaml.

## How to Run

```
bash single_node_slurm.sh
```






