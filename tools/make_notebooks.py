#!/usr/bin/env python3
"""Generates notebooks/colab_{cpu,gpu,tpu}.ipynb. Edit the cell text here, not the .ipynb JSON.

    python tools/make_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip().splitlines(True)}


SETUP = '''
# ---- setup: safe to rerun; also pulls your latest push ----
REPO_URL = "https://github.com/<you>/sgemm.git"   # <- edit me
# Private repo? Store a fine-grained token in Colab Secrets (key icon) named GH_TOKEN, then:
# from google.colab import userdata
# REPO_URL = f"https://{userdata.get('GH_TOKEN')}@github.com/<you>/sgemm.git"

import os
if not os.path.isdir("/content/sgemm"):
    !git clone {REPO_URL} /content/sgemm
%cd /content/sgemm
!git pull --ff-only

# No git yet? Upload sgemm.zip via the Files sidebar instead, then:
# !unzip -qo /content/sgemm.zip -d /content && cd /content/sgemm
'''

PLOTS = '''
# ---- show the plots ----
import glob
from IPython.display import Image, display
for p in sorted(glob.glob("sgemm/{b}/plots/*/*.png")):
    print(p)
    display(Image(p))
'''

SAVE = '''
# ---- keep the results: Colab disk disappears with the session ----
!zip -qr {b}_out.zip sgemm/{b}/results sgemm/{b}/plots
from google.colab import files
files.download("{b}_out.zip")

# Or copy to Drive instead (commit the CSVs to git from your laptop afterwards):
# from google.colab import drive; drive.mount("/content/drive")
# !cp -r sgemm/{b}/results sgemm/{b}/plots /content/drive/MyDrive/sgemm_{b}/
'''


def nb(cells, accelerator=None):
    meta = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    if accelerator:
        meta["accelerator"] = accelerator
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def write(name, notebook):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(notebook, indent=1) + "\n")
    print("wrote", OUT / name)


write("colab_gpu.ipynb", nb([
    md("# SGEMM on a Colab T4\nThin launcher: all logic lives in the repo. Runtime > Change runtime type > **T4 GPU**."),
    code("!nvidia-smi"),
    code(SETUP),
    code("!python run.py gpu info"),
    md("Compile every `sgemm/gpu/kernels/*.cu` with nvcc (sm_75). Rerun after pulling new kernels: only changed files rebuild."),
    code("!python run.py gpu build"),
    md("Smoke test first (`--quick`), then the full sweep. Results are resumable: if the session dies, rerun this cell. "
       "Add `--kernels cublas_fp32,00_naive` to run just some kernels."),
    code("!python run.py gpu run --quick"),
    code("!python run.py gpu run\n!python run.py gpu plot"),
    code(PLOTS.format(b="gpu")),
    code(SAVE.format(b="gpu")),
], accelerator="GPU"))

write("colab_tpu.ipynb", nb([
    md("# SGEMM on a Colab TPU\nThin launcher. Runtime > Change runtime type > **TPU**. "
       "TPU runtimes change over time: use whichever generation Colab currently offers."),
    code("import jax\nprint(jax.__version__, jax.devices())\n"
         "# Must list TPU devices. If it shows CPU: re-pick the TPU runtime and reconnect; if jax is\n"
         "# missing, `!pip install -U \"jax[tpu]\"` then Runtime > Restart session."),
    code(SETUP),
    code("!python run.py tpu info"),
    code("!python run.py tpu run --quick"),
    code("!python run.py tpu run\n!python run.py tpu plot"),
    code(PLOTS.format(b="tpu")),
    code(SAVE.format(b="tpu")),
], accelerator="TPU"))

write("colab_cpu.ipynb", nb([
    md("# SGEMM CPU ladder on a Colab VM (optional second data point next to your laptop)\n"
       "Colab VMs have ~2 shared vCPUs and no cpufreq, so treat absolute GFLOPS as indicative "
       "and compare ratios (percent of OpenBLAS)."),
    code("!apt-get -qq update && apt-get -qq install -y libopenblas-dev"),
    code(SETUP),
    code("!python run.py cpu build\n!python run.py cpu run --quick"),
    code("!python run.py cpu run\n!python run.py cpu plot"),
    code(PLOTS.format(b="cpu")),
    code(SAVE.format(b="cpu")),
]))
