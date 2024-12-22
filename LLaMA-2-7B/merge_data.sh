python3 -m llmshearing.data.merge_data \
        --input_dir /data/LLM-Shearing/llmshearing/data/mds_sample_redpajama \
        --output_dir /data/LLM-Shearing/llmshearing/data/mds_sample_redpajama1 \
        --output_split eval_merge \
        --split_names cc github book stackexchange wiki arxiv c4-rp
