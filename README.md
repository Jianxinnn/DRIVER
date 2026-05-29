<div align="center">

# DRIVER

**A staged zero-shot workflow for ranking protein single-mutant libraries.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

DRIVER enumerates local single-mutant neighborhoods, scores variants with frozen
protein models, and writes ranked candidates for experimental testing. This repository
contains the scoring and ranking code; model weights, wet-lab assays, ESMFold outputs,
MMseqs2 databases, and full ProteinGym benchmarks should be obtained separately.
For Nature Research code/software checklist mapping, see [NATURE_CHECKLIST.md](NATURE_CHECKLIST.md).

## Contents

- [System requirements](#system-requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [Demo data](#demo-data)
- [Two-stage workflow](#two-stage-workflow)
- [Backends](#backends)
- [Outputs](#outputs)
- [Reproducibility notes](#reproducibility-notes)
- [License](#license)

## System requirements

DRIVER is distributed as source code; no compiled standalone binary is required.

Tested author environment for this peer-review package:

| Item | Version/details |
|---|---|
| Operating system | Ubuntu 22.04.5 LTS, Linux 6.8.0 |
| Python | 3.11.11 in the local `mutant` conda environment; Python 3.10+ is supported |
| Python packages | See [requirements.txt](requirements.txt) |
| GPU runtime | NVIDIA driver 550.144.03, CUDA 12.4 |
| Tested GPUs | NVIDIA RTX A6000 / RTX 5880 Ada Generation, 48 GB memory |

No non-standard hardware is required to inspect the code, parse CLI arguments, or use the
bundled input files. A CUDA-capable GPU is recommended for model scoring, and full ensemble
runs that include ESM-2 3B/VespaG are not practical on an ordinary CPU-only desktop.

## Install

Run DRIVER from the repository root.

```bash
git clone https://github.com/Jianxinnn/DRIVER.git
cd DRIVER

conda create -n driver python=3.10 -y
conda activate driver

# Install a PyTorch build matching your CUDA runtime if needed.
# Example: pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Typical install time for the Python environment is 10-30 minutes on a Linux workstation,
excluding pretrained model downloads. The first run of an ESM/ProSST backend can add several
minutes and several GB of network/cache traffic depending on the local Hugging Face cache.

Optional runtime settings:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com   # if your network needs a Hugging Face mirror
```

Install PoET, VespaG, and other external backends only when you run those models.
The package versions used for review/reproduction are pinned in [requirements.txt](requirements.txt);
install commands may need to change for local CUDA and Python versions.

## Quick start

Run the bundled ESM2-only demo:

```bash
python driver_cli.py \
  --fasta examples/esm2_demo.fa \
  --output examples/results/esm2_demo \
  --models esm2 \
  --tasks "single dms" \
  --msas examples/esm2_demo.a3m \
  --gpus 0 \
  --normalization_method 0-1
```

Expected output:

```text
examples/results/esm2_demo/scores/multi_gpu_task_esm2_demo_single_dms_output.csv
examples/results/esm2_demo/esm2_demo.fa_preview.txt
examples/results/esm2_demo/esm2_demo.fa_results.zip
```

For the 10-aa demo sequence, the CSV contains 190 single-mutant rows plus a header. After
ESM2 weights are already cached, the demo is expected to finish in less than 5 minutes on a
CUDA workstation; the first run is dominated by model download time.

See [examples/README.md](examples/README.md) for the staged demo and IscB example files.

## Demo data

Small demo inputs included in this repository:

| File | Purpose |
|---|---|
| `examples/esm2_demo.fa` | 10-aa FASTA sequence for the lightweight ESM2 demo |
| `examples/esm2_demo.a3m` | Matching minimal A3M alignment for the demo |
| `examples/iscb_stage0.csv` / `examples/iscb_stage1.csv` | IscB example output tables |
| `examples/iscb_stage0.pdb` / `examples/iscb_stage1.pdb` | Example IscB structure files |
| `examples/iscb_stage0.a3m` / `examples/iscb_stage1.a3m` | Example IscB MSA files |

To run DRIVER on your own data, provide a FASTA sequence with canonical amino acids. Structure-
based models need a PDB/CIF file or a cached ProSST token file, and MSA-dependent workflows
should use an archived `.a3m` file for repeatable runs.

## Two-stage workflow

At stage `t`, DRIVER starts from seed sequence `x_t`, enumerates single mutants `X_t`,
ranks them with frozen predictor `f_t`, and keeps the top `K_t` candidates.
Measured activity `A(x)` is used only to choose the next seed.

| | Stage 0 | Stage 1 |
|---|---|---|
| Default predictor | `prosst` | `esm2 esmif vaspa poet` |
| Input | FASTA + structure | Stage 0 CSV with one selected/reseedable variant |
| Output | `stage0_top{K}.csv` | `stage1_top{K}.csv` |

**Stage 0**

```bash
python driver_stage.py stage0 \
  --fasta input.fa \
  --output results/stage0_prosst \
  --pdb path/to/structure_or_dir \
  --msas path/to/alignment.a3m \
  --gpus 0 \
  --top_k 20 \
  --normalization_method 0-1
```

**Stage 1**

```bash
python driver_stage.py stage1 \
  --stage0_csv results/stage0_reseed.csv \
  --output results/stage1_ensemble \
  --pdb path/to/stage1_structure_or_dir \
  --msas path/to/alignment.a3m \
  --gpus 0 \
  --top_k 59 \
  --normalization_method 0-1
```

`stage0_reseed.csv` needs `mutseqs` and `mut_info` columns. If it also contains an
activity column, pass `--activity_column activity`; otherwise Stage 1 uses the first row.

Useful flags:

| Flag | Meaning |
|---|---|
| `--top_k` | Candidate budget for the current stage |
| `--models` | Override the default model list |
| `--score_column` | Column used for ranking; defaults to `average` for ensembles |
| `--selection_order` | `desc` by default; use `asc` for lower-is-better scores |
| `--prosst_cache` | Cached ProSST structure tokens instead of on-the-fly PDB quantization |

## Backends

| Backend | CLI name | Requirement |
|---|---|---|
| ESM-2 | `esm2` | Hugging Face `transformers` |
| ESM-1v | `esm1v` | Hugging Face `transformers` |
| ESM-IF | `esmif` | `fair-esm`; requires structure input |
| ProSST | `prosst` | `AI4Protein/ProSST-2048`; requires `--pdb` or `--prosst_cache` |
| VespaG | `vaspa` | `pip install vespag`; set `VESPAG_HOME` / `VESPAG_CHECKPOINT` if needed |
| PoET | `poet` | Set `POET_HOME` and `POET_CHECKPOINT` |

The archived IscB Stage 0 candidates were nominated with a legacy DeProt/early-ProSST checkpoint.

For all MSA-dependent runs, pass `--msas path/to/alignment.a3m` unless MMseqs2 paths are
configured in `config.yaml`.

## Outputs

Stage outputs are written under the directory passed to `--output`.

```text
scores/*_single_dms_output.csv   full single-mutant score table
stage0_top{K}.csv                Stage 0 selected candidates
stage1_seed.fa                   Stage 1 selected seed
stage1_selected_seed.csv         Stage 1 seed metadata
stage1_all_scores.csv            Stage 1 full score table
stage1_top{K}.csv                Stage 1 selected candidates
*_summary.json                   run metadata
*_preview.txt / *_results.zip    lightweight preview and packaged results
```

Generated files under `examples/results/`, `results/`, and `output/` are ignored by Git.

## Reproducibility notes

For manuscript reproduction, provide the seed FASTA files, A3M/MSA files, structure files
or ProSST token caches, command lines, environment file, model identifiers, VespaG/PoET
checkpoint details, ranked CSV outputs, and summary JSON files.

## License

The DRIVER source code is released under the [MIT License](LICENSE). Third-party pretrained
model weights, external databases, and optional backend packages are not redistributed here and
remain subject to their respective licenses.
