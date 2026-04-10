#!/usr/bin/python3
from tqdm import tqdm
from absl import flags, app
from os import environ, listdir
from os.path import join
import gemmi
from concurrent.futures import ProcessPoolExecutor, TimeoutError

FLAGS = flags.FLAGS

def is_pure_protein_cif(cif_path):
    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()

        asym_to_polytype = {}
        for row in block.find(["_struct_asym.id", "_entity_poly.type"]):
            asym_id = row[0]
            poly_type = row[1].strip().lower()
            asym_to_polytype[asym_id] = poly_type

        has_protein = False
        has_nucleic = False

        for pt in asym_to_polytype.values():
            if "polypeptide" in pt:
                has_protein = True
            if "deoxyribonucleotide" in pt or "ribonucleotide" in pt:
                has_nucleic = True

        return has_protein and not has_nucleic

    except Exception:
        return False

def add_options():
    flags.DEFINE_string('output', default='protein_subset.txt', help='输出文件')
    flags.DEFINE_integer('processes', default=16, help='进程数（推荐CPU核心数）')
    flags.DEFINE_float('timeout', default=10.0, help='单个文件超时时间')

def main(unused_argv):
    mmcif_dir = join(environ['PROTENIX_ROOT_DIR'], "mmcif")

    # 收集文件
    file_list = [f for f in listdir(mmcif_dir) if f.endswith('.cif')]
    path_list = [join(mmcif_dir, f) for f in file_list]
    total = len(file_list)
    print(f"✅ 找到 {total} 个 cif 文件，启动多进程筛选...")

    valid_stems = []

    # ========== 多进程核心 ==========
    with ProcessPoolExecutor(max_workers=FLAGS.processes) as executor:
        futures = [executor.submit(is_pure_protein_cif, p) for p in path_list]

        for idx, future in enumerate(tqdm(futures, desc="处理中")):
            try:
                ok = future.result(timeout=FLAGS.timeout)
            except (TimeoutError, Exception):
                ok = False

            if ok:
                valid_stems.append(file_list[idx][:-4])  # 去掉.cif

    # 保存结果
    with open(FLAGS.output, 'w') as f:
        for stem in valid_stems:
            f.write(f"{stem}\n")

    print(f"\n🎉 完成！纯蛋白样本：{len(valid_stems)}")

if __name__ == "__main__":
    add_options()
    app.run(main)
