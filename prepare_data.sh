#!/bin/bash
bash scripts/database/download_pretenix_data.sh --full
python3 scripts/prepare_training_data.py -i sample_list.txt -o preprocessed_indices.csv -b preprocessed_dataset -c clusters-by-entity-40.txt -n 4
