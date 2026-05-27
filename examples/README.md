# Examples

The bundled demo is intentionally lightweight and uses only `esm2`, because no example structure file is included. It checks that mutation generation, scoring, normalization, and staged output wiring work.

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

For real runs, use the README workflow: Stage 0 defaults to ProSST with structure input, and Stage 1 defaults to `esm2 esmif vaspa poet`.
