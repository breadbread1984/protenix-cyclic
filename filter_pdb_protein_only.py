#!/usr/bin/python3

from tqdm import tqdm
from absl import flags, app
from os import environ, listdir
from os.path import join, splitext
import gemmi

FLAGS = flags.FLAGS

def is_pure_protein_cif(cif_path):
  """
  判断CIF是否：
  - 含至少一条多肽链
  - 不含任何DNA/RNA链
  """
  try:
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()

    # 取出 _struct_asym_id 对应的聚合物类型
    # 对应PDBx字段: _entity_poly.type
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

    # 纯蛋白：有蛋白、无核酸
    return has_protein and not has_nucleic

  except Exception as e:
    print(f"⚠️  解析失败 {cif_path}: {e}", file=sys.stderr)
    return False

def add_options():
  flags.DEFINE_string('output', default = 'protein_subset.txt', help = 'path to output file')

def main(unused_argv):
  mmcif_dir = join(environ['PROTENIX_ROOT_DIR'], "mmcif")
  cifs = list()
  for cif in tqdm(listdir(mmcif_dir)):
    stem, ext = splitext(cif)
    if ext != '.cif': continue
    if is_pure_protein_cif(join(mmcif_dir, cif)):
      cifs.append(stem)
  with open(FLAGS.output, 'w') as f:
    for cif in cifs:
      f.write(f"{cif}\n")

if __name__ == "__main__":
  add_options()
  app.run(main)

