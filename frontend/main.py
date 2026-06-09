#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree, copyfile
from os import makedirs, listdir
from os.path import join, exists, splitext, isdir, basename
from functools import partial
import pandas as pd
from datetime import datetime
import threading
import time
import json
from uuid import uuid4
from pathlib import Path
import gradio as gr
from gradio_rangeslider import RangeSlider
from gradio.routes import mount_gradio_app
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from runner import Runner

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'service host')
  flags.DEFINE_integer('port', default = 8082, help = 'service port')
  flags.DEFINE_integer('num_gpus', default = 8, help = 'number of gpu')
  flags.DEFINE_string('output_dir', default = '/root/rfd2_output', help = 'path to output directory')

class Protenix(object):
  def __init__(self, num_gpus):
    self.processes = {gpu_id: None for gpu_id in range(num_gpus)}
    self.status = {gpu_id: "idle" for gpu_id in range(num_gpus)}
    self.logs = {gpu_id: [] for gpu_id in range(num_gpus)}
    self.binder_chain_ids = {}
    self.lock = threading.Lock()
    # start monitor thread
    self.monitor_thread = threading.Thread(target = self._monitor_processes)
    self.monitor_thread.daemon = True
    self.monitor_thread.start()
  def _monitor_processes(self):
    while True:
      with self.lock:
        gpu_ids = list(self.processes.keys())
      for gpu_id in gpu_ids:
        with self.lock:
          process = self.processes.get(gpu_id)
        if process is not None:
          try:
            log, binder_chain_id = next(process)
            with self.lock:
              self.logs[gpu_id].append(log)
              self.binder_chain_ids[gpu_id] = binder_chain_id
          except StopIteration:
            with self.lock:
              self.logs[gpu_id].append(f"{datetime.now()}: process finished")
              self.status[gpu_id] = 'finished'
              self.processes[gpu_id] = None
  def run_protenix(self, pdb_path, 
                         use_msa = False,
                         use_template = False,
                         seed = 1,
                         sample_num = 5,
                         step_num = 200,
                         model = "protenix_base_default_v1.0.0",
                         gpu_id = 0):
    seed = int(seed)
    sample_num = int(sample_num)
    step_num = int(step_num)
    if gpu_id not in self.processes:
      return False, f"invalid GPU ID: {gpu_id}", None
    if self.status[gpu_id] == "running":
      return False, f"GPU {gpu_id} is busy, status: {self.status[gpu_id]}", None
    try:
      output_dir = join(FLAGS.output_dir, str(gpu_id), f'{str(uuid4())}')
      makedirs(output_dir, exist_ok = True)
      runner = Runner()
      with self.lock:
        self.processes[gpu_id] = runner.run(pdb_path, output_dir, use_msa, use_template, seed, sample_num, step_num, model, gpu_id)
        self.status[gpu_id] = 'running'
        self.logs[gpu_id].append(f"{datetime.now()}: start new protenix task")
      return True, f"started task on GPU {gpu_id}, output directory: {output_dir}", output_dir
    except Exception as e:
      return False, f"failed to start protenix: {str(e)}", None
  def get_gpu_status(self, gpu_id):
    return self.status[gpu_id]
  def get_gpu_logs(self, gpu_id):
    if gpu_id in self.logs:
      return '\n'.join(self.logs[gpu_id])
    return "no log"

def create_interface(manager):
  def run_prediction(pdb_path, use_msa, use_template, seed, sample_num, step_num, model, gpu_id):
    if pdb_path is None:
      raise gr.Error("error: please upload file you want to predict with protenix")
    success, message, output_dir = manager.run_protenix(pdb_path, use_msa, use_template, seed, sample_num, step_num, model, gpu_id)
    if success == False:
      raise gr.Error(f'Task failed to run on GPU {gpu_id}, message: {message}')
    else:
      gr.Info(f'Task runs on GPU {gpu_id} successful, message: {message}')
    while True:
      status = manager.get_gpu_status(gpu_id)
      if status in {'finished', 'idle'}:
        yield status, manager.get_gpu_logs(gpu_id), list(), {}
        break
      yield status, manager.get_gpu_logs(gpu_id), list(), {}
      time.sleep(1)

    outputs = list()
    confidences = dict()
    stem, ext = splitext(basename(pdb_path))
    pred_dir = join(output_dir, stem, f"seed_{seed}", "predictions")
    for f in listdir(pred_dir):
      name, ext = splitext(f)
      sn = name.replace(f'{stem}_sample_', '')
      if ext != '.cif': continue
      conf_path = join(pred_dir, f"{stem}_summary_confidence_sample_{sn}.json")
      if not exists(conf_path): continue
      with open(conf_path, 'r') as c:
        confidence = json.loads(c.read())
      outputs.append(join(pred_dir, f))
      confidences[basename(f)] = confidence
    yield manager.get_gpu_status(gpu_id), manager.get_gpu_logs(gpu_id), outputs, confidences

  with gr.Block() as demo:
    with gr.Row():
      gr.Markdown("# Protenix manager tools")
    with gr.Row():
      with gr.Column():
        gr.Markdown("## Submit New Task")
        pdb_input = gr.File(file_types = ['.pdb', '.cif'], type = "filepath", label = "input pdb", interactive = True)
        with gr.Row():
          use_msa = gr.Checkbox(label = "use MSA", value = False)
          use_template = gr.Checkbox(label = "use Template", value = False)
        with gr.Row():
          seed = gr.Number(label = "seed", precision = 0, value = 1, minimum = 1, maximum = 1000)
          sample_num = gr.Number(label = "sample number", precision = 0, value = 5, minimum = 1, maximum = 100)
          step_num = gr.Number(label = "step number", precision = 0, value = 200, minimum = 1, maximum = 10000)
        model = gr.Dropdown(label = "model", choices = ['protenix_base_default_v1.0.0', 'protenix_mini_default_v1.0.0', 'protenix_base_default_v2.0.0', 'protenix_mini_default_v2.0.0'], value = 'protenix_base_default_v1.0.0')
        with gr.Tabs():
          tabs = {device: {'tab': gr.TabItem(f'GPU {device}')} for device in list(manager.logs.keys())}
          for device, widgets in tabs.items():
            with widgets['tab']:
              widgets['column'] = gr.Column()
              with widgets['column']:
                widgets['submit'] = gr.Button('submit prediction on this gpu')
                widgets['status'] = gr.Textbox(label = "status", interactive = False)
                widgets['logs'] = gr.Textbox(label = 'logs', lines = 10, interactive = False)
                widgets['results'] = gr.File(file_count = "multiple", label = "design peptides", interactive = False)
                widgets['confidences'] = gr.JSON(label = "confidences", interactive = False)
      for device, widgets in tabs.items():
        widgets['submit'].click(
          partial(run_prediction, gpu_id = device),
          inputs = [pdb_input, use_msa, use_template, seed, sample_num, step_num, model],
          outputs = [widgets['status'], widgets['logs'], widgets['results'], widgets['confidences']],
          api_name = f"predict_{device}",
        )
    return demo

def main(unused_argv):
  manager = Protenix(FLAGS.num_gpus)
  demo = create_interface(manager)
  application = FastAPI()
  application = mount_gradio_app(app = application, blocks = demo, path = '/')
  uvicorn.run(
    application,
    host = FLAGS.host,
    port = FLAGS.port,
  )

if __name__ == "__main__":
  add_options()
  app.run(main)

