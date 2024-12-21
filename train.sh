torchrun --nproc_per_node=2 main.py \
    --srste_decay 1e-4  \
    --learning_rate 2e-4 \
    --min_lr 2e-5 \
    --teacher_model 'gpt2' \
    --student_model 'gpt2' \
    --warmup_iters 2000 \
    --lr_decay_iters 38000 \
    --max_iters 40000 \
    --batch_size 4\
    --global_batch_size 128\
    --eval_interval 200\
    --mask_metric 'wanda'\
    --change_mask True\
    # --SLoRB True \
    # --SLoRB_init_type "sum"\
    # --SLoRB_k 32 \
    # --trainable_projection True\
    # --wandb_logging True\
    # --distill_model True \
    # --hardness_task 1.0 \
    # --hardness_kldiv 2.0 \
    # --hardness_squarehead 0.0  \
    # --gradient_checkpointing True \






