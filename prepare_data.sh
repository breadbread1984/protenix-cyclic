#!/bin/bash
# recent_pdb_rsync.sh - 2021-09后到2024年底mmCIF (ay→cz)

OUTDIR="./recent_pdb_2021_2024"
mkdir -p "$OUTDIR"

# ========== 2021 Q4 ==========
rsync -rlpt -v -z --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ay/ \
  "$OUTDIR/ay/"

# ========== 2022 全年 (az→bz) ==========
for dir in az ba bb bc bd be bf bg bh bi bj bk bl bm bn bo bp bq br bs bt bu bv bw bx by bz; do
  echo "Syncing 2022: $dir..."
  rsync -rlpt -v -z --delete --port=33444 \
    "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/$dir/" \
    "$OUTDIR/$dir/" &
done
wait

# ========== 2023 全年 (ca→cz) ==========
for dir in ca cb cc cd ce cf cg ch ci cj ck cl cm cn co cp cq cr cs ct cu cv cw cx cy cz; do
  echo "Syncing 2023: $dir..."
  rsync -rlpt -v -z --delete --port=33444 \
    "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/$dir/" \
    "$OUTDIR/$dir/" &
done
wait

# ========== 2024 全年 (da→dz，但截断到dz前) ==========
for dir in da db dc dd de df dg dh di dj dk dl dm dn do dp dq dr ds dt du dv dw dx dy dz; do
  echo "Syncing 2024: $dir..."
  rsync -rlpt -v -z --delete --port=33444 \
    "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/$dir/" \
    "$OUTDIR/$dir/" &
done
wait

echo "✅ 同步完成！2021-2024 PDB数据在 $OUTDIR/"
echo "总计目录：$(ls -d $OUTDIR/* | wc -l)"

find "$OUTDIR" -name '*.cif.gz' -exec gunzip {} \;

find "$OUTDIR" -name *.cif | tr ' ' '\n' > sample_list.txt
python3 scripts/gen_ccd_cache.py -c ccd_cache_dir -n 4
python3 scripts/prepare_training_data.py -i sample_list.txt -o preprocessed_indices.csv -b preprocessed_dataset -c preprocessed_cluster.txt -n 4
