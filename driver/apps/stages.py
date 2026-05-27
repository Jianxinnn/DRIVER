import argparse
import json
from pathlib import Path

import pandas as pd

from driver.core.model_registry import (
    PUBLIC_MODEL_CHOICES,
    STRUCTURE_TOKEN_MODEL,
    normalize_model_names,
)


STAGE0_DEFAULT_MODELS = ["prosst"]
STAGE1_DEFAULT_MODELS = ["esm2", "esmif", "vaspa", "poet"]


def _safe_id(value):
    value = str(value)
    safe = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "sequence"


def _source_id_from_score_path(score_file):
    stem = Path(score_file).stem
    prefix = "multi_gpu_task_"
    suffix = "_single_dms_output"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    return stem


def _resolve_score_column(df, requested=None, models=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Score column '{requested}' not found. Available columns: {list(df.columns)}")
        return requested
    available_models = [model_name for model_name in models or [] if model_name in df.columns]
    if len(available_models) > 1 and "average" in df.columns:
        return "average"
    if len(available_models) == 1:
        return available_models[0]
    if "selected_score" in df.columns:
        return "selected_score"
    if "average" in df.columns:
        return "average"
    if available_models:
        return available_models[-1]
    for model_name in PUBLIC_MODEL_CHOICES:
        if model_name in df.columns:
            return model_name
    raise ValueError(f"No score column found. Available columns: {list(df.columns)}")


def _normalize_models(models):
    return normalize_model_names(models)


def _validate_model_inputs(args):
    if STRUCTURE_TOKEN_MODEL in args.models and not args.pdb and not args.prosst_cache:
        raise ValueError(
            "ProSST scoring requires structure input. "
            "Provide --pdb with a PDB/CIF file or directory, or --prosst_cache."
        )
    if "esmif" in args.models and not args.pdb and not args.predict_structure:
        raise ValueError(
            "ESM-IF scoring requires structure input. Provide --pdb, or add --predict_structure "
            "to predict structures for the input sequences."
        )


def _select_top(df, score_column, top_k, selection_order):
    sort_columns = [score_column]
    ascending = [selection_order == "asc"]
    if "mut_info" in df.columns:
        sort_columns.append("mut_info")
        ascending.append(True)
    return df.sort_values(sort_columns, ascending=ascending, kind="mergesort").head(top_k).copy()


def _dms_score_files(result_files):
    return [
        Path(path)
        for path in result_files
        if Path(path).name.endswith("_single_dms_output.csv")
    ]


def _run_scoring(args, fasta_path, output_dir):
    from driver.apps.cli import run_multi_gpu, set_seed

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    return run_multi_gpu(
        fasta_file=str(fasta_path),
        output_dir=str(output_dir),
        MODEL_USES=args.models,
        predict_structure=args.predict_structure,
        task_types=["single dms"],
        pdb_file=args.pdb,
        prosst_pdb_cache=args.prosst_cache,
        gpus=args.gpus,
        seed=args.seed,
        msas=args.msas,
        normalization_method=args.normalization_method,
    )


def run_stage0(args):
    output_dir = Path(args.output).resolve()
    fasta_path = Path(args.fasta).resolve()

    zip_path, preview_path, heatmap_paths, result_files = _run_scoring(args, fasta_path, output_dir)
    score_files = _dms_score_files(result_files)
    if not score_files:
        raise FileNotFoundError("No Stage 0 score CSV files were produced.")

    top_frames = []
    for score_file in score_files:
        df = pd.read_csv(score_file)
        score_column = _resolve_score_column(df, args.score_column, args.models)
        top_df = _select_top(df, score_column, args.top_k, args.selection_order)
        top_df.insert(0, "source_id", _source_id_from_score_path(score_file))
        top_df.insert(1, "selected_score_column", score_column)
        top_df.insert(2, "selected_score", top_df[score_column])
        top_frames.append(top_df)

    top_path = output_dir / f"stage0_top{args.top_k}.csv"
    pd.concat(top_frames, ignore_index=True).to_csv(top_path, index=False)

    summary = {
        "stage": 0,
        "stage_index_t": 0,
        "seed_symbol": "x_0",
        "candidate_set_symbol": "X_0",
        "predictor_symbol": "f_0",
        "budget_symbol": "K_0",
        "meaning": "Scan all single mutants from the input seed sequence(s).",
        "fasta": str(fasta_path),
        "models": args.models,
        "normalization_method": args.normalization_method,
        "selection_order": args.selection_order,
        "K_t": args.top_k,
        "score_files": [str(path) for path in score_files],
        "top_candidates": str(top_path),
        "zip": str(zip_path),
        "preview": str(preview_path),
        "heatmaps": [str(path) for path in heatmap_paths],
    }
    with open(output_dir / "stage0_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Stage 0 completed: {top_path}")


def _write_stage1_seed_fasta(row, output_dir, sequence_column, mutation_column):
    fasta_path = output_dir / "stage1_seed.fa"
    mutation = row.get(mutation_column, "seed")
    if pd.isna(mutation):
        mutation = "seed"
    seq_id = _safe_id(f"stage1_seed_{mutation}")
    sequence = str(row[sequence_column])

    with open(fasta_path, "w") as handle:
        handle.write(f">{seq_id}\n{sequence}\n")

    row_dict = row.to_dict()
    row_dict["stage1_seed_id"] = seq_id
    selected_path = output_dir / "stage1_selected_seed.csv"
    pd.DataFrame([row_dict]).to_csv(selected_path, index=False)
    return fasta_path, selected_path


def _select_stage1_seed(stage0_df, args):
    if stage0_df.empty:
        raise ValueError(f"No rows found in {args.stage0_csv}.")

    if args.activity_column:
        if args.activity_column not in stage0_df.columns:
            raise ValueError(
                f"Activity column '{args.activity_column}' not found in {args.stage0_csv}. "
                f"Available columns: {list(stage0_df.columns)}"
            )
        selected = _select_top(stage0_df, args.activity_column, 1, args.activity_order).iloc[0]
        return selected, args.activity_column, "activity_column"

    return stage0_df.iloc[0].copy(), None, "csv_order"


def run_stage1(args):
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage0_df = pd.read_csv(args.stage0_csv)
    if args.sequence_column not in stage0_df.columns:
        raise ValueError(f"Sequence column '{args.sequence_column}' not found in {args.stage0_csv}.")

    selected, activity_column, reseed_mode = _select_stage1_seed(stage0_df, args)

    seed_fasta, selected_path = _write_stage1_seed_fasta(
        selected,
        output_dir,
        args.sequence_column,
        args.mutation_column,
    )

    zip_path, preview_path, heatmap_paths, result_files = _run_scoring(args, seed_fasta, output_dir)
    score_files = _dms_score_files(result_files)
    if not score_files:
        raise FileNotFoundError("No Stage 1 score CSV files were produced.")
    if len(score_files) != 1:
        raise RuntimeError("Stage 1 expects exactly one reseeded sequence.")

    score_file = score_files[0]
    combined_df = pd.read_csv(score_file)
    combined_df.insert(0, "stage1_seed_id", _source_id_from_score_path(score_file))
    combined_path = output_dir / "stage1_all_scores.csv"
    combined_df.to_csv(combined_path, index=False)

    score_column = _resolve_score_column(combined_df, args.score_column, args.models)
    top_df = _select_top(combined_df, score_column, args.top_k, args.selection_order)
    top_df.insert(1, "selected_score_column", score_column)
    top_df.insert(2, "selected_score", top_df[score_column])
    top_path = output_dir / f"stage1_top{args.top_k}.csv"
    top_df.to_csv(top_path, index=False)

    summary = {
        "stage": 1,
        "stage_index_t": 1,
        "seed_symbol": "x_1",
        "candidate_set_symbol": "X_1",
        "predictor_symbol": "f_1",
        "budget_symbol": "K_1",
        "activity_symbol": "A(x)",
        "meaning": "Use the single reseeded sequence selected after Stage 0 and scan one additional mutation around it.",
        "stage0_csv": str(Path(args.stage0_csv).resolve()),
        "reseed_mode": reseed_mode,
        "activity_column": activity_column,
        "activity_order": args.activity_order,
        "stage1_score_column": score_column,
        "selection_order": args.selection_order,
        "K_t": args.top_k,
        "seed_fasta": str(seed_fasta),
        "selected_seed": str(selected_path),
        "score_files": [str(path) for path in score_files],
        "combined_scores": str(combined_path),
        "top_candidates": str(top_path),
        "zip": str(zip_path),
        "preview": str(preview_path),
        "heatmaps": [str(path) for path in heatmap_paths],
    }
    with open(output_dir / "stage1_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Stage 1 completed: {top_path}")


def _add_scoring_args(parser, default_models):
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help=(
            "Models to use. Available: "
            f"{', '.join(PUBLIC_MODEL_CHOICES)}. "
            f"Default for this stage: {' '.join(default_models)}."
        ),
    )
    parser.add_argument("--predict_structure", action="store_true", help="Predict structure for structure-based models")
    parser.add_argument("--pdb", type=str, help="Path to PDB/CIF file or structure directory")
    parser.add_argument(
        "--prosst_cache",
        dest="prosst_cache",
        type=str,
        help="Path to cached ProSST structure tokens",
    )
    parser.add_argument("--gpus", nargs="+", default=[0], type=int, help="GPU IDs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--msas", type=str, help="Path to existing MSA file")
    parser.add_argument(
        "--normalization_method",
        type=str,
        default="0-1",
        choices=["minmax", "0-1", "zscore", "z-score"],
        help="Score normalization method (default: 0-1)",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Two-stage DRIVER workflow")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    stage0 = subparsers.add_parser("stage0", help="Scan single mutants from input seed sequence(s)")
    stage0.add_argument("--fasta", required=True, help="Input FASTA file for Stage 0")
    stage0.add_argument("--output", default="results/stage0", help="Stage 0 output directory")
    stage0.add_argument("--top_k", "--K_t", dest="top_k", type=int, default=20, help="Number of Stage 0 candidates to keep (K_0)")
    stage0.add_argument("--score_column", help="Column used to select top candidates")
    stage0.add_argument(
        "--selection_order",
        choices=["asc", "desc"],
        default="desc",
        help="Use desc for larger-is-better scores; use asc only for lower-is-better custom scores",
    )
    _add_scoring_args(stage0, STAGE0_DEFAULT_MODELS)
    stage0.set_defaults(func=run_stage0)

    stage1 = subparsers.add_parser("stage1", help="Scan mutations around one reseeded Stage 1 sequence")
    stage1.add_argument("--stage0_csv", required=True, help="Stage 0 score, top-candidate, or reseed CSV")
    stage1.add_argument("--output", default="results/stage1", help="Stage 1 output directory")
    stage1.add_argument(
        "--activity_column",
        "--reseed_score_column",
        dest="activity_column",
        help="Experimental activity column A(x) used to choose x_1 from --stage0_csv",
    )
    stage1.add_argument(
        "--activity_order",
        choices=["asc", "desc"],
        default="desc",
        help="Use desc when higher experimental activity is better",
    )
    stage1.add_argument("--top_k", "--K_t", dest="top_k", type=int, default=20, help="Number of Stage 1 candidates to keep after scoring X_1 (K_1)")
    stage1.add_argument("--score_column", help="Score column used to rank Stage 1 candidates; defaults to the ensemble average")
    stage1.add_argument("--sequence_column", default="mutseqs", help="Column containing seed sequences")
    stage1.add_argument("--mutation_column", default="mut_info", help="Column containing mutation labels")
    stage1.add_argument(
        "--selection_order",
        choices=["asc", "desc"],
        default="desc",
        help="Use desc for larger-is-better scores; use asc only for lower-is-better custom scores",
    )
    _add_scoring_args(stage1, STAGE1_DEFAULT_MODELS)
    stage1.set_defaults(func=run_stage1)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.models = _normalize_models(args.models)
        _validate_model_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.func(args)
