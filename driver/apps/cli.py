import os
from driver.core.runtime import configure_runtime_environment, load_runtime_config

configure_runtime_environment()

import argparse
import logging
import queue
import random
import subprocess
import sys
import zipfile

from driver.core.model_registry import PUBLIC_MODEL_CHOICES, normalize_model_names


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_gpu(gpu_id):
    import torch

    try:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            logger.info(f"Successfully set up GPU {gpu_id} as device {device}")
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')
    except Exception as e:
        logger.error(f"Error setting up GPU {gpu_id}: {e}")
        raise

def read_fasta(fasta_file):
    from Bio import SeqIO

    sequences = []
    try:
        # Count sequences before validation.
        total = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
        logger.info(f"Found {total} sequences in FASTA file")
        
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            
            # Validate sequence content strictly.
            if not seq:
                logger.warning(f"Empty sequence found for {record.id}")
                continue
                
            if len(seq) < 10:
                logger.warning(f"Sequence too short for {record.id}: {len(seq)}")
                continue
                
            if not all(c in valid_aas for c in seq):
                invalid_chars = set(seq) - valid_aas
                logger.warning(f"Invalid characters in {record.id}: {invalid_chars}")
                continue
                
            sequences.append((record.id, seq))
            logger.info(f"Processed sequence {record.id}: length={len(seq)}")
            
        if not sequences:
            raise ValueError("No valid sequences found in FASTA file")
            
        logger.info(f"Successfully loaded {len(sequences)} valid sequences")
        return sequences
        
    except Exception as e:
        logger.error(f"Error reading FASTA file: {e}")
        raise

config = load_runtime_config()


def process_single_sequence(
    seq_id,
    sequence,
    task_name,
    MODEL_USES,
    predict_structure,
    task_types,
    pdb_file,
    prosst_pdb_cache,
    score_dir,
    seed=42,
    msas=None,
    device=None,
    normalization_method="0-1",
):
    import pandas as pd

    from driver.core.scoring import ProteinMutator, esm2_vocabs

    MODEL_USES = normalize_model_names(MODEL_USES)
    current_task_name = f"{task_name}_{seq_id}"
    
    pdb_path = config['pdb_path']
    msas_path = config['msas_path']
    os.makedirs(score_dir, exist_ok=True)

    a3m_file_path = os.path.join(msas_path, f"{current_task_name}.a3m")
    pdb_file_path = os.path.join(pdb_path, f"pdb/{current_task_name}.pdb") \
        if os.path.exists(os.path.join(pdb_path, f"pdb/{current_task_name}.pdb")) else None

    if pdb_file:
        # If a directory is provided, find a PDB/CIF file matching the FASTA sequence ID.
        if os.path.isdir(pdb_file):
            candidate_paths = [
                os.path.join(pdb_file, f"{seq_id}.pdb"),
                os.path.join(pdb_file, f"{seq_id}.PDB"),
                os.path.join(pdb_file, f"{seq_id}.cif"),
                os.path.join(pdb_file, f"{seq_id}.CIF"),
                os.path.join(pdb_file, f"{seq_id.lower()}.pdb"),
                os.path.join(pdb_file, f"{seq_id.lower()}.cif"),
                os.path.join(pdb_file, f"{seq_id.upper()}.pdb"),
                os.path.join(pdb_file, f"{seq_id.upper()}.cif"),
            ]
            selected_pdb = next((p for p in candidate_paths if os.path.exists(p)), None)
            if selected_pdb:
                pdb_file_path = selected_pdb
                logger.info(f"Using structure file {selected_pdb} for {seq_id} from directory {pdb_file}")
            else:
                raise FileNotFoundError(f"No PDB/CIF file found for {seq_id} in directory {pdb_file}")
        else:
            # A single PDB/CIF file is shared by all input sequences.
            pdb_file_path = pdb_file
            logger.info(f"Using provided single structure file {pdb_file} for {seq_id}")
    elif pdb_file_path is None and predict_structure:
        fa_dir = os.path.join(pdb_path, "fa")
        output_pdb_dir = os.path.join(pdb_path, "pdb")
        os.makedirs(fa_dir, exist_ok=True)
        os.makedirs(output_pdb_dir, exist_ok=True)
        fasta_for_structure = os.path.join(fa_dir, f"{current_task_name}.fa")
        with open(fasta_for_structure, "w") as f:
            f.write(f">{current_task_name}\n{sequence}")
        logger.info(f"Predicting structure for {seq_id}")
        command = ["conda", "run", "-n", "structure", "esm-fold", "-i", fasta_for_structure, "-o", output_pdb_dir]
        subprocess.run(command, check=True)
        pdb_file_path = os.path.join(output_pdb_dir, f"{current_task_name}.pdb")
    elif not predict_structure:
        logger.info(f"Not predicting structure for {seq_id}")
        pdb_file_path = None

    if msas:
        logger.info(f"Using provided MSA file for {seq_id}")
        a3m_file_path = msas
    elif not os.path.exists(a3m_file_path):
        logger.info(f'Running MMseqs2 for {seq_id}')
        command = [
            sys.executable,
            "-m",
            "driver.data.msa_search",
            "--wt",
            sequence,
            "--entry",
            current_task_name,
            "--outdir",
            msas_path,
        ]
        subprocess.run(command, check=True)
        a3m_file_path = os.path.join(msas_path, f"{current_task_name}.a3m")
    else:
        logger.info(f"Using existing MSA file for {seq_id}: {a3m_file_path}")


    models_dict = {
        model: None
        for model in [
            'esm1v',
            'esm2',
            'esmif',
            'vaspa',
            'poet',
            'prosst',
            'prosst_tokenizer',
            'esm2_tokenizer',
            'esmif_alphabet_',
            'esm_model_3b',
        ]
    }
    models_dict['MODEL_USES'] = MODEL_USES

    protein_mutator = ProteinMutator(model_config=models_dict, device=device or 'cuda', seed=seed)
    logger.info(f"Sequence {seq_id} using device {protein_mutator.device}")
    protein_mutator._load_models()

    result_files = []
    preview_texts = []
    heatmap_paths = []

    if "single dms" in task_types:
        dms_output = protein_mutator.run_dms(
            wt=sequence, 
            pdb2speicific=pdb_file_path, 
            msa2speicific=a3m_file_path, 
            out_name=current_task_name,
            scaler=True,
            prosst_pdb_cache=prosst_pdb_cache,
            normalization_method=normalization_method,
        )
        dms_output_csv_path = os.path.join(score_dir, f"{current_task_name}_single_dms_output.csv")
        dms_output.to_csv(dms_output_csv_path, index=False)
        result_files.append(dms_output_csv_path)
        preview_model_columns = [model for model in MODEL_USES if model in dms_output.columns]
        if len(preview_model_columns) > 1 and "average" in dms_output.columns:
            preview_column = "average"
        elif preview_model_columns:
            preview_column = preview_model_columns[0]
        else:
            preview_column = "average"
        top_20_output = dms_output.sort_values(by=preview_column, ascending=False).head(20).to_string(index=False)
        preview_texts.append(f"Single DMS Results for {seq_id}:\n" + top_20_output)
    
    if "single mask recovery" in task_types:
        import matplotlib.pyplot as plt
        import seaborn as sns

        recovery_output = protein_mutator.run_single_mask(seq=sequence, pdb=pdb_file_path, out_name=current_task_name, model_list=MODEL_USES)
        for model_name, logits in recovery_output.items():
            logits_df = pd.DataFrame(logits)
            logits_df.columns = esm2_vocabs
            recovery_output_csv_path = os.path.join(
                score_dir,
                f"{current_task_name}_{model_name}_single_mask_recovery_output.csv",
            )
            logits_df.to_csv(recovery_output_csv_path, index=True)
            result_files.append(recovery_output_csv_path)

            plt.figure(figsize=(10, 8))
            sns.heatmap(logits_df, cmap="viridis")
            plt.title(f"{model_name} Reconstructed Logits Heatmap for {seq_id}")
            plt.xlabel("Position")
            plt.ylabel("Amino Acid")
            heatmap_path = os.path.join(score_dir, f"{current_task_name}_{model_name}_heatmap.png")
            plt.savefig(heatmap_path)
            heatmap_paths.append(heatmap_path)
            plt.close()

    return result_files, preview_texts, heatmap_paths


def process_gpu_batch(
    gpu_id,
    sequences,
    task_name,
    MODEL_USES,
    predict_structure,
    task_types,
    pdb_file,
    prosst_pdb_cache,
    result_queue,
    score_dir,
    seed=42,
    msas=None,
    normalization_method="0-1",
):
    import torch

    MODEL_USES = normalize_model_names(MODEL_USES)
    device = setup_gpu(gpu_id)
    
    # Monitor GPU memory.
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated(device)
        logger.info(f"Initial GPU {gpu_id} memory: {initial_memory/1024**2:.2f}MB")
    
    batch_result_files = []
    batch_preview_texts = []
    batch_heatmap_paths = []
    
    # Dynamic batch size.
    batch_size = min(32, len(sequences))

    try:
        for i in range(0, len(sequences), batch_size):
            sub_sequences = sequences[i:i+batch_size]
            for seq_id, sequence in sub_sequences:
                logger.info(f"GPU {gpu_id}: Processing {seq_id}")
                results = process_single_sequence(
                    seq_id, sequence, task_name, MODEL_USES,
                    predict_structure, task_types, pdb_file, prosst_pdb_cache,
                    score_dir, seed, msas, device, normalization_method
                )
                batch_result_files.extend(results[0])
                batch_preview_texts.extend(results[1])
                batch_heatmap_paths.extend(results[2])

            # Periodically clear the GPU cache.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated(device)
                logger.info(f"Current GPU {gpu_id} memory: {current_memory/1024**2:.2f}MB")

        result_queue.put({
            "ok": True,
            "result_files": batch_result_files,
            "preview_texts": batch_preview_texts,
            "heatmap_paths": batch_heatmap_paths,
        })
    except Exception as exc:
        logger.exception("GPU %s failed", gpu_id)
        result_queue.put({"ok": False, "error": repr(exc)})
        raise

def run_multi_gpu(
    fasta_file,
    output_dir,
    MODEL_USES,
    predict_structure,
    task_types,
    pdb_file,
    prosst_pdb_cache=None,
    gpus=None,
    seed=42,
    msas=None,
    normalization_method="0-1",
):
    import torch

    MODEL_USES = normalize_model_names(MODEL_USES)
    processes = []
    try:
        output_dir = os.path.abspath(output_dir)
        score_dir = os.path.join(output_dir, "scores")
        os.makedirs(score_dir, exist_ok=True)

        sequences_to_process = list(read_fasta(fasta_file))
        logger.info(f"Processing {len(sequences_to_process)} sequences")
        
        if not gpus:
            num_gpus = torch.cuda.device_count()
            gpus = list(range(num_gpus))
            if num_gpus == 0:
                logger.warning("No GPUs found. Running on CPU.")
                gpus = [0]
        
        # Optimize task assignment.
        num_gpus = len(gpus)
        num_sequences = len(sequences_to_process)
        
        # Use no more GPUs than the number of sequences.
        if num_gpus > num_sequences:
            gpus = gpus[:num_sequences]
            num_gpus = len(gpus)
            logger.info(f"Adjusted number of GPUs to {num_gpus} based on sequence count")
        
        # Assign work based on sequence length.
        sequences_per_gpu = [[] for _ in range(num_gpus)]
        seq_lengths = [(i, len(seq[1])) for i, seq in enumerate(sequences_to_process)]
        seq_lengths.sort(key=lambda x: x[1], reverse=True)  # Sort by length descending.
        
        # Greedily balance sequence length across GPUs.
        gpu_loads = [0] * num_gpus
        for seq_idx, length in seq_lengths:
            # Find the least loaded GPU.
            min_load_gpu = min(range(num_gpus), key=lambda x: gpu_loads[x])
            sequences_per_gpu[min_load_gpu].append(sequences_to_process[seq_idx])
            gpu_loads[min_load_gpu] += length
            
        # Validate assignment.
        for i, seqs in enumerate(sequences_per_gpu):
            logger.info(f"GPU {gpus[i]} assigned {len(seqs)} sequences, "
                       f"total length: {sum(len(seq[1]) for seq in seqs)}")
            if not seqs:  # Warn if a GPU has no assigned sequences.
                logger.warning(f"GPU {gpus[i]} has no sequences assigned, skipping...")
                continue

        if num_gpus == 1:
            result_queue = queue.Queue()
            process_gpu_batch(
                gpus[0],
                sequences_per_gpu[0],
                'multi_gpu_task',
                MODEL_USES,
                predict_structure,
                task_types,
                pdb_file,
                prosst_pdb_cache,
                result_queue,
                score_dir,
                seed,
                msas,
                normalization_method,
            )
            message = result_queue.get()
            if not message.get("ok"):
                raise RuntimeError(message.get("error", "GPU batch failed"))
            all_result_files = message["result_files"]
            all_preview_texts = message["preview_texts"]
            all_heatmap_paths = message["heatmap_paths"]
        else:
            # Start one process per GPU with assigned work.
            import torch.multiprocessing as mp

            mp.set_start_method('spawn', force=True)
            result_queues = []

            for i, (gpu_id, seqs) in enumerate(zip(gpus, sequences_per_gpu)):
                if seqs:  # Start processes only for non-empty GPU assignments.
                    result_queue = mp.Queue()
                    process = mp.Process(
                        target=process_gpu_batch,
                        args=(gpu_id, seqs, 'multi_gpu_task', MODEL_USES,
                              predict_structure, task_types, pdb_file, prosst_pdb_cache,
                              result_queue, score_dir, seed, msas, normalization_method)
                    )
                    processes.append(process)
                    result_queues.append(result_queue)
                    process.start()
                    logger.info(f"Started process on GPU {gpu_id}")

            # Collect results.
            all_result_files = []
            all_preview_texts = []
            all_heatmap_paths = []

            for process, result_queue in zip(processes, result_queues):
                process.join()
                try:
                    message = result_queue.get(timeout=1)
                except queue.Empty as exc:
                    raise RuntimeError(f"GPU process {process.pid} produced no result message") from exc
                if not message.get("ok"):
                    raise RuntimeError(message.get("error", f"GPU process {process.pid} failed"))
                if process.exitcode != 0:
                    raise RuntimeError(f"GPU process {process.pid} exited with code {process.exitcode}")
                all_result_files.extend(message["result_files"])
                all_preview_texts.extend(message["preview_texts"])
                all_heatmap_paths.extend(message["heatmap_paths"])

        # Save results.
        if not all_result_files:
            raise RuntimeError("No result files were produced.")
        
        zip_path = os.path.join(output_dir, f"{os.path.basename(fasta_file)}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in all_result_files:
                zipf.write(file, os.path.basename(file))

        preview_path = os.path.join(output_dir, f"{os.path.basename(fasta_file)}_preview.txt")
        with open(preview_path, 'w') as f:
            f.write("\n\n".join(all_preview_texts))

        logger.info(f"Processing completed. Results saved to {zip_path}")
        return zip_path, preview_path, all_heatmap_paths, all_result_files

    except Exception as e:
        logger.error(f"Error in multi-GPU processing: {e}")
        raise
    finally:
        # Clean up resources.
        torch.cuda.empty_cache()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        logger.info("Cleaned up all resources")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Protein Mutant Zero-shot Platform")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument(
        "--models",
        nargs='+',
        default=['esm2'],
        help=(
            "Models to use. Available: "
            f"{', '.join(PUBLIC_MODEL_CHOICES)}."
        ),
    )
    parser.add_argument("--predict_structure", action="store_true", help="Predict structure (only needed for esmif model)")
    parser.add_argument("--tasks", nargs='+', default=['single dms'], choices=['single dms', 'single mask recovery'], help="Tasks to perform")
    parser.add_argument(
        "--pdb",
        type=str,
        help=(
            "Path to PDB/CIF file or directory (optional). "
            "If a directory is provided, structure files named as FASTA IDs "
            "(e.g. seq1.pdb or seq1.cif) will be used per sequence."
        ),
    )
    parser.add_argument(
        "--prosst_cache",
        dest="prosst_cache",
        type=str,
        help="Path to cached ProSST structure tokens (optional)",
    )
    parser.add_argument("--gpus", nargs='+', default=[0,1], type=int, help="GPU IDs to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--msas", type=str, help="Path to existing MSA files")
    parser.add_argument(
        "--normalization_method",
        type=str,
        default="0-1",
        choices=["minmax", "0-1", "zscore", "z-score"],
        help="Score normalization method when integrating model outputs (default: 0-1)",
    )
    
    args = parser.parse_args()
    try:
        args.models = normalize_model_names(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    set_seed(args.seed)
    zip_path, preview_path, heatmap_paths, result_files = run_multi_gpu(
        fasta_file=args.fasta,
        output_dir=args.output,
        MODEL_USES=args.models,
        predict_structure=args.predict_structure,
        task_types=args.tasks,
        pdb_file=args.pdb,
        prosst_pdb_cache=args.prosst_cache,
        gpus=args.gpus,
        seed=args.seed,
        msas=args.msas,
        normalization_method=args.normalization_method,
    )

    logger.info(f"Processing completed. Results are in {zip_path} and {preview_path}")

if __name__ == "__main__":
    main()
