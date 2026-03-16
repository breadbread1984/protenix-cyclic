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
from pathlib import Path
import gradio as gr
from gradio.routes import mount_gradio_app
from fastapi import FastAPI
from fastapi.responses import FileResponse
import threading
import subprocess
from predictor import Predictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('num_gpus', default = 4, help = 'number of gpus')
  flags.DEFINE_string("service_host", default = "0.0.0.0", help = 'service host')
  flags.DEFINE_integer("service_port", default = 8081, help = 'service port')
  flags.DEFINE_string('input_dir', default = '/root/px_input', help = 'path to input directory')
  flags.DEFINE_string('output_dir', default = '/root/px_output', help = 'path to output directory')
  flags.DEFINE_enum("model", enum_values = {
    'protenix_base_default_v1.0.0',
    'protenix_base_20250630_v1.0.0',
    'protenix_base_default_v0.5.0',
    'protenix_base_constraint_v0.5.0',
    'protenix_mini_esm_v0.5.0',
    'protenix_mini_ism_v0.5.0',
    'protenix_mini_default_v0.5.0',
    'protenix_tiny_default_v0.5.0',
  }, default = "protenix_base_default_v1.0.0", help = "model to use")
  flags.DEFINE_string('ccd_dir', default = '/root/ccd', help = 'ccd directory')

class ProtenixManager(object):
  def __init__(self, num_gpus):
    self.processes = {gpu_id: None for gpu_id in range(num_gpus)}
    self.status = {gpu_id: "idle" for gpu_id in range(num_gpus)}
    self.logs = {gpu_id: [] for gpu_id in range(num_gpus)}
    self.lock = threading.Lock()
    # start monitor thread
    self.monitor_thread = threading.Thread(target = self._monitor_processes)
    self.monitor_thread.daemon = True
    self.monitor_thread.start()
    # download ccd
    assert exists(FLAGS.ccd_dir)
  def _monitor_processes(self):
    while True:
      for gpu_id, process in self.processes.items():
        if process is not None:
          try:
            log = next(process)
            with self.lock:
              self.logs[gpu_id].append(log)
          except StopIteration:
            with self.lock:
              self.logs[gpu_id].append(f"{datetime.now()}: process finished")
              self.status[gpu_id] = 'finished'
              self.processes[gpu_id] = None
      #time.sleep(1)
  def run_protenix(self, gpu_id, input_file, seed = 1):
    if gpu_id not in self.processes:
      return False, f"invalid GPU ID: {gpu_id}"
    if self.status[gpu_id] == 'running':
      return False, f"GPU {gpu_id} is busy, status: {self.status[gpu_id]}"
    try:
      makedirs(join(FLAGS.output_dir, str(gpu_id)), exist_ok = True)
      px = Predictor(FLAGS.model, FLAGS.input_dir, join(FLAGS.output_dir, str(gpu_id)))
      with self.lock:
        self.processes[gpu_id] = px.predict(
          input_file,
          seed = seed,
          gpu_id = gpu_id,
        )
        self.status[gpu_id] = 'running'
        self.logs[gpu_id].append(f"{datetime.now()}: start new protenix task")
      return True, f"started task on GPU {gpu_id}, output directory: {join(FLAGS.output_dir, str(gpu_id))}"
    except Exception as e:
      return False, f"failed to start Protenix: {str(e)}"
  def get_gpu_status(self, gpu_id):
    return self.status[gpu_id]
  def get_gpu_logs(self, gpu_id):
    if gpu_id in self.logs:
      return '\n'.join(self.logs[gpu_id])
    return "no log"

def create_interface(manager):
  with gr.Blocks(title = "AlphaFold3 manager") as interface:
    # 1) interface
    with gr.Row():
      gr.Markdown("# AlphaFold3 manager tools")
    with gr.Row():
      with gr.Tab("Task Manager"):
        with gr.Column():
          with gr.Row():
            gr.Markdown('### Submit New Task')
          with gr.Row(equal_height = True):
            input_file = gr.File(label = 'input file', file_types = ['.fasta', '.cif'])
          with gr.Row():
            seed = gr.Number(label = 'seed', value = 1, minimum = 1, precision = 0)
          with gr.Row():
            gr.Markdown('### Device Selector')
          with gr.Row():
            with gr.Tabs('device selector') as device_selector:
              tabs = {device: {'tab': gr.TabItem(f'GPU {device}')} for device in list(manager.logs.keys())}
              for device, widgets in tabs.items():
                with widgets['tab']:
                  widgets['column'] = gr.Column()
                  with widgets['column']:
                    widgets['submit'] = gr.Button('submit prediction on this gpu')
                    widgets['status'] = gr.Textbox(label = "status", interactive = False)
                    widgets['logs'] = gr.Textbox(label = 'logs', lines = 10, interactive = False)

      with gr.Tab('Result Visualization') as view_tab:
        with gr.Column():
          gr.Markdown('### Prediction Viewer')
          view_gpu_selector = gr.Dropdown(
            choices = list(manager.processes.keys()),
            label = 'GPU selection',
            value = list(manager.processes.keys())[0] if len(manager.processes) else None
          )
          with gr.Row():
            download = gr.File(label = 'download')
            view_btn = gr.Button(value = 'visualize')
          results = gr.Dataframe(headers = ['job name', 'path'], datatype = ['str', 'str'], interactive = False)
    # 2) callbacks
    def run_prediction(input_file, seed, gpu_id):
      if input_file is None:
        raise gr.Error("error: please upload file you want to prediction")
      success, message = manager.run_protenix(
        gpu_id = int(gpu_id),
        input_file = input_file,
        seed = seed
      )
      if success == False:
        raise gr.Error(f'Task failed to run on GPU {gpu_id}, message: {message}')
      else:
        gr.Info(f'Task runs on GPU {gpu_id} successful, message: {message}')
      while True:
        yield manager.get_gpu_status(gpu_id), manager.get_gpu_logs(gpu_id)
        if manager.get_gpu_status(gpu_id) in {'finished', 'idle'}: break
        time.sleep(1)
    def list_results(gpu_id):
      output_dir = join(FLAGS.output_dir, str(gpu_id))
      cifs = list()
      if exists(output_dir):
        for res_dir in listdir(output_dir):
          job_dir = join(output_dir, res_dir)
          if not isdir(job_dir): continue
          cifs.extend([join(job_dir, f) for f in listdir(job_dir) if splitext(f)[1] == '.cif'])
      return gr.update(value = pd.DataFrame([{'job name': splitext(basename(cif))[0][:-6], 'path': cif} for cif in cifs]))
    def prepare_files(df, evt: gr.SelectData):
      row_index = evt.index[0]
      clicked_row_values = evt.row_value
      sample_path = clicked_row_values[1]
      with open(sample_path, 'r') as f:
        cif_content = f.read()
      copyfile(sample_path, 'selected.cif')
      html = f"""<!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>NGL Viewer</title>
            <!-- 引入NGL Viewer库 -->
            <script src="https://cdn.jsdelivr.net/npm/ngl@0.10.4/dist/ngl.js"></script>
            <style>
                #viewport {{ width: 100%; height: 500px; }}
            </style>
        </head>
        <body>
            <div id="viewport"></div>
            <script>
                // 创建查看器实例
                var stage = new NGL.Stage("viewport");
                content = `{cif_content}`;
                // 定义加载PDB内容的函数
                var stringBlob = new Blob([content], {{type: 'text/plain'}});
                stage.loadFile(stringBlob, {{ext: "cif", defaultRepresentation: true}});
            </script>
        </body>
    </html>
        """
      with open('selected.html', 'w') as f:
        f.write(html)
      return 'selected.cif'
    def py_openwindows():
      # dummy callback
      pass
    js_openwindows = """
    () => {
        // 创建新窗口
        window.open('/selected.html', '_blank');
        return true;
    }
    """
    # 3) events
    for device, widgets in tabs.items():
      widgets['submit'].click(
        partial(run_prediction, gpu_id = device),
        inputs = [input_file, seed],
        outputs = [widgets['status'], widgets['logs']]
      )
    view_tab.select(
      list_results,
      inputs = [view_gpu_selector],
      outputs = [results]
    )
    view_gpu_selector.change(
      list_results,
      inputs = [view_gpu_selector],
      outputs = [results]
    )
    results.select(
      prepare_files,
      inputs = [results],
      outputs = [download]
    )
    view_btn.click(
      fn = py_openwindows,
      inputs = None,
      outputs = None,
      js = js_openwindows
    )
  return interface

application = FastAPI()

@application.get("/selected.html")
def selected():
  return FileResponse("selected.html")

def main(unused_argv):
  gr.set_static_paths(paths = [Path(FLAGS.output_dir)])
  global application
  import uvicorn
  manager = ProtenixManager(FLAGS.num_gpus)
  interface = create_interface(manager)
  application = mount_gradio_app(app = application, blocks = interface, path = '/')
  uvicorn.run(
    application,
    host = FLAGS.service_host,
    port = FLAGS.service_port
  )

if __name__ == "__main__":
  add_options()
  app.run(main)
