#!/bin/bash
export PROTENIX_ROOT_DIR=/home/xieyi/release_data
bash scripts/database/download_pretenix_data.sh --full
# inference once to download pretrained model
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
# generate finetuning list
python3 filter_pdb_protein_only.py --output filename_list.txt --processes 32 --timeout 300
"""
python3 scripts/prepare_training_data.py \
  -i mmcif \
  -o indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz \
  -b mmcif_bioassembly \
  -c common/clusters-by-entity-40.txt \
  -n 64 \
  -d
"""
