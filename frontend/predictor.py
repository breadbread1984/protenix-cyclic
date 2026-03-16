#!/usr/bin/python3

from shutil import 
from os import mkdir, environ
from os.path import abspath, dirname, exists, join, splitext, isdir, basename
import tempfile
import json
import subprocess

class Predictor(object):
  def __init__(self, model, input_dir, output_dir,):
    self.model = model
    self.input_dir = input_dir
    self.output_dir = output_dir # output directory ends with gpu_id
  def tojson(self, input_file):
    # run converter
    proc = subprocess.Popen([
      'protenix',
      'json',
      '--input',
      input_file,
      '--out_dir',
      self.input_dir,
      stdout = subprocess.PIPE,
      stderr = subprocess.STDOUT,
      text = True,
      bufsize = 1,
      universal_newlines = True,
    ])
    # wait the process to end
    try:
      while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
          break
    except:
      proc.kill()
    # return converted file's path
    stem, _ = splitext(input_file)
    input_file = join(self.input_dir, f"{stem[:20]}.json") # NOTE: https://github.com/bytedance/Protenix/blob/main/runner/batch_inference.py#L1001
    return input_file
  def predict(self, input_file, seed = 1, gpu_id = 0):
    assert exists(self.input_dir) and isdir(self.input_dir)
    assert exists(self.output_dir) and isdir(self.output_dir)
    stem, ext = splitext(input_file)
    json_file = self.tojson(input_file)
    env = environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(gpu_id)})
    proc = subprocess.Popen(
      [
        'protenix',
        'pred',
        '-i',
        json_file,
        '-o',
        self.output_dir,
        '-n',
        self.model,
        '-s',
        str(seed),
        '--use_template',
        True,
        '--use_default_params',
        True
      ],
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
    except:
      proc.kill()

