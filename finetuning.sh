#!/bin/bash

python3 runner/train.py \
--run_name protenix_finetune \
--seed 101 \
--base_dir ./ckpt_20260319 \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 48 \
--eval_interval 1000 \
--log_interval 50 \
--eval_first false \
--checkpoint_interval 2000 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 1000000 \
--warmup_steps 100 \
--lr 0.0001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path /home/xiebo/checkpoint/protenix_base_default_v1.0.0.pt \
--load_ema_checkpoint_path /home/xiebo/checkpoint/protenix_base_default_v1.0.0.pt \
--data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list examples/finetune_subset.txt \
--data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
