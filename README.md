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

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Two-stage workflow](#two-stage-workflow)
- [Backends](#backends)
- [Outputs](#outputs)
- [License](#license)

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

Optional runtime settings:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com   # if your network needs a Hugging Face mirror
```

Install PoET, VespaG, and other external backends only when you run those models.

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

See [examples/README.md](examples/README.md) for the staged demo and IscB example files.

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

## License

Released under the [MIT License](LICENSE).
