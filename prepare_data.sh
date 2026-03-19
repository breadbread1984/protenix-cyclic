#!/bin/bash
wget https://af3-dev.tos-cn-beijing.volces.com/release_data.tar.gz
tar -xzvf release_data.tar.gz && rm release_data.tar.gz
python3 scripts/prepare_training_data.py -i sample_list.txt -o preprocessed_indices.csv -b preprocessed_dataset -c clusters-by-entity-40.txt -n 4
