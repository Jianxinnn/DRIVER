# Examples

The bundled demo is intentionally lightweight and uses only `esm2`, so it does not require a
structure file, ProSST, ESM-IF, VespaG, PoET, or external checkpoints. It checks mutation
generation, scoring, normalization, and staged output wiring.

```bash
conda run -n mutant python driver_cli.py \
  --fasta examples/esm2_demo.fa \
  --output examples/results/esm2_demo \
  --models esm2 \
  --tasks "single dms" \
  --msas examples/esm2_demo.a3m \
  --gpus 0 \
  --normalization_method 0-1
```

Expected outputs:

- `examples/results/esm2_demo/scores/multi_gpu_task_esm2_demo_single_dms_output.csv`
- `examples/results/esm2_demo/esm2_demo.fa_preview.txt`
- `examples/results/esm2_demo/esm2_demo.fa_results.zip`

For the 10-aa demo sequence, the CSV contains 190 single-mutant rows plus a header.
After ESM2 weights are cached, expected runtime is less than 5 minutes on a CUDA workstation;
the first run may take longer because the model weights must be downloaded.

The same small input can also test the stage wrapper by overriding the stage defaults to `esm2`:

```bash
conda run -n mutant python driver_stage.py stage0 \
  --fasta examples/esm2_demo.fa \
  --output examples/results/stage0_demo \
  --models esm2 \
  --msas examples/esm2_demo.a3m \
  --gpus 0 \
  --top_k 3 \
  --normalization_method 0-1

conda run -n mutant python driver_stage.py stage1 \
  --stage0_csv examples/results/stage0_demo/stage0_top3.csv \
  --output examples/results/stage1_demo \
  --models esm2 \
  --msas examples/esm2_demo.a3m \
  --gpus 0 \
  --top_k 3 \
  --normalization_method 0-1
```

Expected staged outputs include:

- `examples/results/stage0_demo/stage0_top3.csv`
- `examples/results/stage0_demo/stage0_summary.json`
- `examples/results/stage1_demo/stage1_seed.fa`
- `examples/results/stage1_demo/stage1_selected_seed.csv`
- `examples/results/stage1_demo/stage1_all_scores.csv`
- `examples/results/stage1_demo/stage1_top3.csv`
- `examples/results/stage1_demo/stage1_summary.json`

For real runs, use the README workflow: Stage 0 defaults to ProSST with structure input, and Stage 1 defaults to `esm2 esmif vaspa poet`.
