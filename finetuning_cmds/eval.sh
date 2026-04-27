#!/bin/bash

torchrun --nproc_per_node=1 runner/train.py \
        --model_name protenix_base_default_v1.0.0 \
        --max_steps 0 \
        --load_checkpoint_path <to be filled> \
        --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925 \
        --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
        --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list filename_list.txt \
        --base_dir ./eval_output \
        --run_name eval_only \
        --dtype bf16 \
        --log_interval 1 \
        --eval_interval 1
