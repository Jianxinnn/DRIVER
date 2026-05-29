# Nature Code and Software Checklist Mapping

Status date: 2026-05-29

This file maps the DRIVER repository contents to the Nature Research Code and Software
Submission Checklist. It is intended for peer review packaging; replace repository/archive
placeholders with final DOI and release metadata before submission.

## Required content

| Checklist item | DRIVER location/status |
|---|---|
| Source code | Python source under `driver/`, `prosst/`, `driver_cli.py`, and `driver_stage.py` |
| Small dataset to demo the software | `examples/esm2_demo.fa` and `examples/esm2_demo.a3m` |
| System requirements | `README.md` > System requirements |
| Dependency versions | `requirements.txt` |
| Tested versions | `README.md` > System requirements and `requirements.txt` |
| Non-standard hardware | `README.md` > System requirements |
| Installation instructions | `README.md` > Install |
| Typical installation time | `README.md` > Install |
| Demo instructions | `README.md` > Quick start and `examples/README.md` |
| Expected demo output | `README.md` > Quick start and `examples/README.md` |
| Expected demo runtime | `README.md` > Quick start and `examples/README.md` |
| How to run on user data | `README.md` > Demo data and Two-stage workflow |
| Reproduction instructions | `README.md` > Reproducibility notes; archive exact manuscript inputs/outputs separately |
| License | `LICENSE` and `README.md` > License |
| Open repository link | `https://github.com/Jianxinnn/DRIVER` |
| Code functionality description | `README.md` > Two-stage workflow; cite the manuscript Methods section for full pseudocode |

## Items to complete before submission

- Tag the exact manuscript code version and record the commit SHA.
- Create a permanent code archive DOI, for example through Zenodo, Figshare, or Code Ocean.
- Deposit manuscript-scale FASTA files, A3M/MSA files, structure files or ProSST token caches,
  command scripts, ranked output tables, summary JSON files, and benchmark outputs.
- Record model identifiers and VespaG/PoET checkpoint details used for the submitted analyses.
- Record exact runtime measurements for each manuscript run on the final hardware.
- Replace author-identifying repository or contact details as needed for double-blind review.
