#!/usr/bin/python3

from os import environ, mkdir
from os.path import splitext, join, exists
import subprocess

class Runner(object):
  def __init__(self,):
    pass
  def tojson(self, pdb_path,):
    stem, ext = splitext(pdb_path)
    tmp_dir = "/tmp"
    out_json = join(tmp_dir, f'{stem}.json')
    cmds = [
      "protenix",
      "json",
      "--input",
      pdb_path,
      "--out_dir",
      tmp_dir,
      "--altloc",
      "first",
      "--include_discont_poly_poly_bonds",
    ]
    proc = subprocess.Popen(
      cmds,
      stdout = subprocess.PIPE,
      stderr = subprocess.STDOUT,
      text = True,
      bufsize = 1,
      universal_newlines = True,
    )
    try:
      while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
          break
        if output:
          print(output.strip())
    except Exception:
      proc.kill()
      return False, None
    return True, out_json
  def run(self, pdb_path,
                output_dir,
                use_msa = False,
                use_template = False,
                seed = 1,
                sample_num = 5,
                step_num = 200,
                model = "protenix_base_default_v1.0.0",
                gpu_id = 0):
    succeed, input_path = self.tojson(pdb_path)
    if succeed == False:
      yield "convert pdb to json failed!", None
      return
    if not exists(output_dir): mkdir(output_dir)
    env = environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cmds = [
      "protenix",
      "pred",
      "-i",
      input_path,
      "-o",
      output_dir,
      "-s",
      str(seed),
      "-n",
      model,
      *(["--use_template"] if use_template else []),
      *(["--use_msa"] if use_msa else []),
      "--sample_diffusion.N_sample",
      str(sample_num),
      "--sample_diffusion.N_step",
      str(step_num),
    ]
    proc = subprocess.Popen(
      cmds,
      env = env,
      stdout = subprocess.PIPE,
      stderr = subprocess.STDOUT,
      text = True,
      bufsize = 1,
      universal_newlines = True,
    )
    try:
      while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
          break
        if output:
          yield output.strip()
    except Exception:
      proc.kill()

