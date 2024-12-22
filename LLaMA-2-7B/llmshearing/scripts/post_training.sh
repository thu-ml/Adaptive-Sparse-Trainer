#!/bin/bash

module load anaconda/2024.02
module load cuda/12.1
module load nccl/1.3.4-cuda12.2
source activate llama
conda activate llama

https_proxy=http://u-LDSRbU:Pq56RrRZ@10.255.128.108:3128
http_proxy=http://u-LDSRbU:Pq56RrRZ@10.255.128.108:3128

PROJ_DIR=/data/home/huangweiyu/LLM-Shearing
DATA_DIR=/ssd/hwy/datapreprocessing/for_prune
OUTPUT_DIR=/data/home/huangweiyu/LLM-Shearing/llmshearing/output
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
TRAIN_SCRIPT=${PROJ_DIR}/train.py

test=True

model=7b
to_model=7b
config_file=${PROJ_DIR}/llmshearing/configs/llama2/${model}.yaml
prune_run_name=llama2_7b_pruning_scaling_doremi_to${model}_sl4096
path=/data/home/huangweiyu/LLM-Shearing/models/Llama-2-7b-composer/state_dict.pt
# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=8
global_train_batch_size=128
device_eval_batch_size=8

# learning setup
lr=1e-5 # learning rate for the main parameters
max_duration=15000ba # 50B tokens
save_interval=1500ba # save every 3200ba
t_warmup=500ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp]  # domain names
# proportion=[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845] # final proportion of pruning
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] #initial proportion of pruning
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
# if [[ $to_model == 1.3b ]]; then
#     target_loss=[1.9643,0.7459,2.1393,1.6117,1.7590,1.4449,2.1251] # 1.3b predicted loss from scaling law
# else
#     target_loss=[1.8712,0.6883,2.0325,1.5353,1.6297,1.3560,2.0328] # 2.7b predicted loss from scaling law
# fi
target_loss=[1.7520,0.6216,1.9063,1.4482,1.4844,1.2637,1.9219]
eval_split_name=eval_merge # eval on all domains
eval_interval=100ba # eval every 100 batches and update the loading proportion

# save directroy
run_name=${prune_run_name}_ft${max_duration}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-01:00:00; else t=01-00:00:00; fi

composer  -n=8 $TRAIN_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=true \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    model.l0_module=null \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=true 
