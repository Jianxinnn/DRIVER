import os
from driver.core.runtime import configure_runtime_environment

configure_runtime_environment()

import logging
import pandas as pd
import numpy as np
import torch
from transformers import  EsmModel, EsmConfig, EsmForMaskedLM,EsmTokenizer
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter
import pickle
from Bio import SeqIO
import torch.nn.functional as F
from tqdm import tqdm
import math
from driver.core.mutations import find_mutations
from driver.core.model_registry import normalize_model_names
from driver.models.loading import load_models
from driver.core.constants import index_map

import gc
from driver.core.normalization import check_torch2numpy, handle_empty_lists, normalize_data_sets
from copy import deepcopy
from driver.core.mutations import generate_mutations, find_mutant_for_prosst
from prosst.structure.quantizer import PdbQuantizer
import random

logger = logging.getLogger(__name__)

esm2_vocabs = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']


class ProteinMutator:
    def __init__(self, model_config, device='cuda', batch_size=32, dropout=False, seed=42):
        self.seed = seed
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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

        self.model_config = model_config
        self.model_config['MODEL_USES'] = normalize_model_names(self.model_config.get('MODEL_USES', []))
        self.esmif_alphabet = model_config['esmif_alphabet_']
        self.esm2_tokenizer = model_config['esm2_tokenizer']
        self.esm2_vocabs = esm2_vocabs
        self.esm2_model = model_config['esm2']
        self.esm_model_3b = model_config['esm_model_3b']
        self.model_if1 = model_config['esmif']
        self.vaspag_model = model_config['vaspa']
        self.poet = model_config['poet']
        self.esm1v_model = model_config['esm1v']
        self.prosst = model_config.get('prosst')
        self.prosst_tokenizer = model_config.get('prosst_tokenizer')

        self.batch_size = batch_size    
        self.device = device
        self.dropout = dropout

    def _load_models(self):
        # Load models and update the shared model dictionary.
        if 'esm1v' in self.model_config['MODEL_USES']:
            self.esm1v_model, self.esm2_tokenizer = load_models('facebook/esm1v_t33_650M_UR90S_1', device=self.device)
            self.model_config['esm1v'] = self.esm1v_model.eval()

        if 'esm2' in self.model_config['MODEL_USES']:
            self.esm2_model, self.esm2_tokenizer = load_models('facebook/esm2_t33_650M_UR50D', device=self.device)
            self.model_config['esm2'] = self.esm2_model.eval()

        if 'esmif' in self.model_config['MODEL_USES']:
            self.model_if1, self.esmif_alphabet = load_models(model_if1_name='esm_if1_gvp4_t16_142M_UR50', device=self.device)
            self.model_config['esmif'] = self.model_if1.eval()
            self.model_config['esmif_alphabet_'] = self.esmif_alphabet
            self.esm2_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

        if 'vaspa' in self.model_config['MODEL_USES']:
            self.esm_model_3b, self.vaspag_model = load_models(vaspag_model=True, device=self.device)
            self.model_config['esm_model_3b'] = self.esm_model_3b.eval()
            self.model_config['vaspa'] = self.vaspag_model
            self.esm2_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

        if 'poet' in self.model_config['MODEL_USES']:
            self.poet = load_models(poet=True, device=self.device)
            self.model_config['poet'] = self.poet

        if 'prosst' in self.model_config['MODEL_USES']:
            self.prosst, self.prosst_tokenizer = load_models(prosst=True, device=self.device)
            self.model_config['prosst'] = self.prosst.eval()
            self.model_config['prosst_tokenizer'] = self.prosst_tokenizer
            self.vocab_prosst = self.prosst_tokenizer.get_vocab()

    def _offload_models(self, model_attr_names):
        """Move selected models from GPU to CPU to release GPU memory."""
        for attr_name in model_attr_names:
            model = getattr(self, attr_name, None)
            if model is not None and hasattr(model, 'cpu'):
                model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_seqs(self, is_dms=False, pdb2dms: str=None, is_group_muts=False, ipt_seq=None):
        # Load sequence data.
        if is_dms:
            if pdb2dms is None and ipt_seq is None:
                raise ValueError
            elif pdb2dms is not None:
                coords, native_seq = esm.inverse_folding.util.load_coords(pdb2dms, 'A')
                mutant_sequence, _ = generate_mutations(native_seq)
                return mutant_sequence, native_seq, [pdb2dms], coords
            elif ipt_seq is not None:
                mutant_sequence, _ = generate_mutations(ipt_seq)
                return mutant_sequence, ipt_seq, None, None
            else:
                raise ValueError
        else:
            if pdb2dms is not None:
                coords, native_seq = esm.inverse_folding.util.load_coords(pdb2dms, 'A')
                return native_seq, native_seq, [pdb2dms], coords
            raise ValueError

    def esmif_logits_batch(self, pdbs):
        # Get ESM-IF predictions.
        with torch.no_grad():
            batch_data = []
            for pdb in pdbs:
                coord, na_seq = esm.inverse_folding.util.load_coords(pdb, 'A')
                batch_data.append((coord, None, na_seq))

            # batch_data = [(coord, None, na_seq) for pdb in pdbs for coord, na_seq in zip(esm.inverse_folding.util.load_coords(pdb, 'A'))]
            batch_converter = CoordBatchConverter(self.esmif_alphabet)
            coords_, confidence, strs, tokens, padding_mask = batch_converter(batch_data)
            prev_output_tokens = tokens[:, :-1]
            target = tokens[:, 1:]
            if_logits, extra = self.model_if1.forward(coords_.to(self.device), padding_mask.to(self.device), confidence.to(self.device), prev_output_tokens.to(self.device))
            softmax_if = torch.nn.functional.softmax(if_logits.permute(1, 0, 2), dim=-1)
            return [strs, if_logits]

    def vaspag_logits_batch(self, seqs):
        # Get VasPaG predictions.
        with torch.no_grad():
            esm_tk = self.esm2_tokenizer(seqs, return_tensors="pt", padding=False, truncation=True)
            esm_tk = {k: v.to(self.device) for k, v in esm_tk.items()}
            esm_ot = self.esm_model_3b(**esm_tk, output_hidden_states=True)
            vaspa_g_logits = self.vaspag_model(esm_ot.hidden_states[-1])
            vaspa_g_logits = vaspa_g_logits[:, 1:-1, :]
            batch_logits = [logits[:, index_map] for logits in vaspa_g_logits]
            return batch_logits

    def _get_prosst_structure_tokens(self, pdb_file: str):
        processor = PdbQuantizer()
        structure_sequence = processor(pdb_file)
        if structure_sequence is None or len(structure_sequence) == 0:
            raise ValueError(f"ProSST structure quantization produced empty tokens for: {pdb_file}. "
                             f"Ensure the structure file matches the sequence, contains backbone atoms, and is a valid PDB/CIF.")
        structure_sequence_offset = [i + 3 for i in structure_sequence]
        self.structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
        # with open(f'output/cache/prosst_structure_cache/{self.task_name}.pkl', 'wb') as f:
        #     pickle.dump(self.structure_input_ids, f)
        self.structure_input_ids = self.structure_input_ids.to(self.device)

    def prosst_logits_batch(self, variant, pdb_file=None, pdb_cache=None):
        logger.debug("ProSST structure source: %s", pdb_file or f"cache:{pdb_cache}")
        if pdb_cache is not None:
            # Robustly load cached structure tokens regardless of how they were saved
            try:
                cached = torch.load(pdb_cache, map_location='cpu')
            except Exception:
                with open(pdb_cache, 'rb') as f:
                    cached = pickle.load(f)
            if isinstance(cached, torch.Tensor):
                self.structure_input_ids = cached.to(self.device)
            else:
                self.structure_input_ids = torch.tensor(cached, dtype=torch.long).to(self.device)
        elif pdb_file is not None:
            self._get_prosst_structure_tokens(pdb_file)
        else:
            raise ValueError("Please provide pdb file or cache.")

        with torch.no_grad():
            tokenized_res = self.prosst_tokenizer([variant], return_tensors='pt')
            input_ids = tokenized_res['input_ids'].to(self.device)
            attention_mask = tokenized_res['attention_mask'].to(self.device)
            # Defensive check: structure tokens length must match sequence tokens length
            seq_len = int(input_ids.size(1))
            ss_len = int(self.structure_input_ids.size(1))
            if ss_len != seq_len:
                src = pdb_file if pdb_file is not None else f"cache:{pdb_cache}"
                raise ValueError(
                    f"ProSST input length mismatch (seq tokens={seq_len}, structure tokens={ss_len}). "
                    f"Likely structure parsing/quantization failed or structure doesn't match sequence length. Source={src}"
                )
            outputs = self.prosst(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=self.structure_input_ids
            )

        logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)
        # logits = [logit[:, index_map] for logit in logits]
        return logits[0]
        
    def esm2_logits_batch(self, seqs, dropout=False):
        # Get ESM-2 predictions.
        with torch.no_grad():
            esm_tk = self.esm2_tokenizer(seqs, return_tensors="pt", padding=False, truncation=True, max_length=min(map(len, seqs)) + 2)
            esm_tk = {k: v.to(self.device) for k, v in esm_tk.items()}
            esm_ot = self.esm2_model(**esm_tk)
            esm_logits = F.softmax(esm_ot.logits, dim=-1)[:, 1:-1, 4:24]
            batch_logits = [logits for logits in esm_logits]
            return batch_logits

    def process_vaspa_batch(self, mut_ids, wt_vaspa_logits, batch_seqs):
        # Process VasPaG predictions.
        vaspa_scores = []
        for mut_id in mut_ids:
            # ensure tensor accumulation on the same device as logits
            if isinstance(wt_vaspa_logits, torch.Tensor):
                vaspa_score = torch.tensor(0.0, device=wt_vaspa_logits.device)
            else:
                # e.g., numpy array
                vaspa_score = 0.0
            for i in mut_id:
                vaspa_score = vaspa_score + wt_vaspa_logits[i[0], i[1]]
                
            # sigmoid in torch if tensor, else in numpy/math
            if isinstance(vaspa_score, torch.Tensor):
                vaspa_score = torch.sigmoid(vaspa_score).item()
            else:
                vaspa_score = 1.0 / (1.0 + math.exp(-float(vaspa_score)))
            vaspa_scores.append([float(vaspa_score)])
        return vaspa_scores
    
    def process_prosst_batch(self, mut_ids, prosst_logits):
        pred_scores = []
        for mutant in mut_ids:

            mutant_score = 0
            for sub_mutant in mutant.split(":"):
                if not sub_mutant:   
                    logger.debug("Empty ProSST mutation entry, skipping")
                    continue
                if len(sub_mutant) < 3:
                    raise ValueError(f"Invalid sub_mutant format: {sub_mutant}")
                try:
                    wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                    pred = prosst_logits[idx, self.vocab_prosst[mt]] - prosst_logits[idx, self.vocab_prosst[wt]]
                    mutant_score += pred.cpu().detach().numpy().item()
                except Exception as e:
                    raise ValueError(f"Error processing sub_mutant {sub_mutant}: {e}") from e
            pred_scores.append([mutant_score])

        return pred_scores

    def process_esmif_batch(self, batch_seqs, if_logits_list, native_seq):
        # Process ESM-IF predictions.
        esmif_scores = []
        if len(if_logits_list) == 1:
            if_logits_list = [if_logits_list[0] for _ in range(len(batch_seqs))]
            native_seqs = [native_seq for _ in range(len(batch_seqs))]

        for if_logits, mt in zip(if_logits_list, batch_seqs):
            esm_tk = self.esm2_tokenizer(mt, return_tensors="pt")
            esm_tk = {k: v.to(if_logits.device) for k, v in esm_tk.items()}
            losses_if = -F.cross_entropy(if_logits.unsqueeze(0), esm_tk['input_ids'][:, 1:-1], reduction='none').mean().cpu().detach().numpy().item()
            esmif_scores.append([losses_if])
        return esmif_scores

    def process_esmif_batch_dropout(self, batch_seqs, if_logits_list, native_seq):
        # Process ESM-IF predictions with dropout.
        esmif_scores = []
        for if_logits, mt in zip(if_logits_list, batch_seqs):
            esm_tk = self.esm2_tokenizer(mt, return_tensors="pt")
            esm_tk = {k: v.to(if_logits.device) for k, v in esm_tk.items()}
            losses_if = -F.cross_entropy(if_logits.unsqueeze(0), esm_tk['input_ids'][:, 1:-1], reduction='none').mean().cpu().detach().numpy().item()
            esmif_scores.append([losses_if])
        return esmif_scores

    def process_esm2_batch(self, batch_seqs, mut_ids, wt, msa=None, dropout=False, wt_margin=False):
        # Process ESM-2 predictions.
        esm2_margins = []
        esm2_lls = []

        def calculate_margin(esm_logits, mut_id):
            # Calculate the ESM-2 margin score.
            margin = sum(
                torch.log(esm_logits[i[0], i[1]]) - torch.log(esm_logits[i[0], i[2]]) for i in mut_id)
            return margin
        
        def calculate_ll(esm_logits, batch_seqs, mut_id):
            logits = esm_logits.unsqueeze(0).permute(0, 2, 1)  # [batch_size, num_classes, seq_length]
            targets = self.esm2_tokenizer(batch_seqs, return_tensors="pt")['input_ids'][:, 1:-1].to(esm_logits.device) - 4  # [batch_size, seq_length]
            assert torch.all(targets >= 0) and torch.all(targets < 21), "Targets contain invalid class indices"
            loss = F.cross_entropy(logits, targets, reduction='none')
            return -loss.mean().cpu().detach().numpy().item()
        
        if wt_margin:
            wt_logits = self.esm2_logits_batch([wt])[0]
            for mut_id in mut_ids:
                wt_margin_value = calculate_margin(wt_logits, mut_id)
                esm2_margins.append([wt_margin_value.cpu().detach().numpy().item()])
            return esm2_margins
        else:
            batch_esm_results = self.esm2_logits_batch(batch_seqs)
            for mt_logit, mut_seq, mut_id in zip(batch_esm_results, batch_seqs, mut_ids):
                esm2_ll = calculate_ll(mt_logit, mut_seq, mut_id)
                esm2_lls.append([esm2_ll])
            return esm2_lls

    def single_dms(self, wt, pdb,
                   MSA_seqs=None,
                   mutant_sequence=None,
                   if_model_dropout=False,
                   poet_msa_np=None,
                   poet_msa_seqs=None,
                   pdb_cache=None,
                ):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Run single-mutant scanning.
        vaspa_results = []
        esm2_results = []
        esmif_results = []
        poet_results = []
        esm1v_results = []
        prosst_results = []

        # Phase 1: compute WT logits, then offload models to CPU to release GPU memory.
        # VASPA uses an ESM-3B backbone and benefits most from immediate offloading.
        if self.vaspag_model is not None:
            wt_vaspa_logits = self.vaspag_logits_batch(wt)[0]
            wt_vaspa_logits = wt_vaspa_logits.cpu()
            self._offload_models(['esm_model_3b', 'vaspag_model'])

        # ESM-IF
        if self.model_if1 is not None:
            esmif_wt_logits = self.esmif_logits_batch(pdb)[1]
            esmif_wt_logits = esmif_wt_logits.cpu()
            self._offload_models(['model_if1'])

        # ProSST
        if self.prosst is not None:
            prosst_pdb_file = pdb[0] if pdb is not None else None
            prosst_wt_logits = self.prosst_logits_batch(wt, pdb_file=prosst_pdb_file, pdb_cache=pdb_cache)
            prosst_wt_logits = prosst_wt_logits.cpu()
            self._offload_models(['prosst'])

        # Phase 2: process all PoET variants once to avoid repeated MSA embedding work.
        if self.poet is not None:
            from driver.models.poet import alphabet as poet_alphabet
            from driver.models.poet import append_startstop, get_poet_predict

            all_variants = [
                append_startstop(poet_alphabet.encode(bytes(v, 'utf-8')), alphabet=poet_alphabet)
                for v in mutant_sequence
            ]
            all_poet_scores = get_poet_predict(
                self.poet, poet_msa_seqs, all_variants, poet_msa_np,
                max_tokens=122880, max_similarity=0.95, batch_size=2, seed=self.seed
            )
            poet_results = list(all_poet_scores)
            self._offload_models(['poet'])

        # Phase 3: batch loop; only ESM2/ESM1v stay on GPU.
        mutants_infos = []
        for i in tqdm(range(0, len(mutant_sequence), self.batch_size)):
            batch_seqs = mutant_sequence[i:i+self.batch_size]
            mut_ids = [find_mutations(wt, mt, self.esm2_vocabs) for mt in batch_seqs]
            mut_prosst_info = [find_mutant_for_prosst(wt, mt) for mt in batch_seqs]

            mutants_infos.extend(mut_ids)

            if self.vaspag_model is not None:
                vaspa_results.extend(self.process_vaspa_batch(mut_ids, wt_vaspa_logits, batch_seqs))
            if self.prosst is not None:
                prosst_results.extend(self.process_prosst_batch(mut_prosst_info, prosst_wt_logits))
            if self.model_if1 is not None:
                if if_model_dropout:
                    esmif_results.extend(self.process_esmif_batch_dropout(batch_seqs, esmif_wt_logits, wt))
                else:
                    esmif_results.extend(self.process_esmif_batch(batch_seqs, esmif_wt_logits, wt))
            if self.esm2_model is not None:
                esm2_results.extend(self.process_esm2_batch(batch_seqs, mut_ids, wt, MSA_seqs))
            if self.esm1v_model is not None:
                esm1v_results.extend(self.process_esm2_batch(batch_seqs, mut_ids, wt, MSA_seqs))

        return vaspa_results, prosst_results, esm2_results, esmif_results, poet_results, esm1v_results, mutants_infos

    def run_dms(
        self,
        wt,
        pdb2speicific,
        msa2speicific,
        out_name,
        scaler=False,
        prosst_pdb_cache=None,
        testing=False,
        normalization_method="0-1",
    ):
        # Run single-mutant scanning.
        poet_msa_np = None
        poet_msa_seqs = None
        if self.poet is not None:
            msa_fasta_path_aligned = msa2speicific
            if msa_fasta_path_aligned is None:
                raise ValueError("MSA path is None. Ensure MMseqs2 finished or provide --msas path.")
            from driver.models.poet import alphabet as poet_alphabet
            from driver.models.poet import get_encoded_msa_from_a3m_seqs

            msa_seqs_aligned = {}
            for record in SeqIO.parse(msa_fasta_path_aligned, "fasta"):
                msa_seqs_aligned[record.id] = bytes(str(record.seq), "utf-8")
            poet_msa_np = get_encoded_msa_from_a3m_seqs(msa_sequences=msa_seqs_aligned.values(), alphabet=poet_alphabet)[:]
            poet_msa_seqs = list(msa_seqs_aligned.values())[:]

        mutant_sequence, native_seq, pdb, coords = self.load_seqs(is_dms=True, pdb2dms=pdb2speicific, ipt_seq=wt)
        results_name = out_name if out_name is not None else ValueError
        logger.info("Running DMS scoring for %s (sequence length=%d)", results_name, len(native_seq))
        logger.debug("DMS inputs: msa=%s pdb=%s", msa2speicific, pdb2speicific)

        vaspa_res, prosst_res, esm2_res, esmif_res, poet_res, esm1v_res, mutants_infos = self.single_dms(
            native_seq,
            pdb,
            mutant_sequence=mutant_sequence,
            poet_msa_np=poet_msa_np,
            poet_msa_seqs=poet_msa_seqs,
            pdb_cache=prosst_pdb_cache
        )

        logger.info(
            "Score counts for %s: esm2=%d esmif=%d vaspa=%d poet=%d esm1v=%d prosst=%d",
            results_name,
            len(esm2_res),
            len(esmif_res),
            len(vaspa_res),
            len(poet_res),
            len(esm1v_res),
            len(prosst_res),
        )


        non_empty_results = {
            'esm2': [check_torch2numpy(x) for x in esm2_res] if len(esm2_res) > 0 else [],
            'esmif': [check_torch2numpy(x) for x in esmif_res] if len(esmif_res) > 0 else [],
            'vaspa': [check_torch2numpy(x) for x in vaspa_res] if len(vaspa_res) > 0 else [],
            'prosst': [check_torch2numpy(x) for x in prosst_res] if len(prosst_res) > 0 else [],
            'poet': [check_torch2numpy(x).reshape([-1,1]) for x in poet_res] if len(poet_res) > 0 else [],
            'esm1v': [check_torch2numpy(x) for x in esm1v_res] if len(esm1v_res) > 0 else [],
        }

        mut_infos = [','.join([j[-1] for j in i]) for i in mutants_infos]
        mut_poss = [[int(j[0]) + 1 for j in i] for i in mutants_infos]
        if scaler:
            normalize_data_sets(
                [non_empty_results[key] for key in non_empty_results if len(non_empty_results[key]) > 0],
                method=normalization_method,
            )
        data_shape = len(mutants_infos)

        dms_results = {
            'mut_info': mut_infos,
            'mutseqs': handle_empty_lists(mutant_sequence, data_shape),
            'mut_pos': mut_poss
        }

        for key, value in non_empty_results.items():
            if len(value) > 0:
                dms_results[key] = handle_empty_lists([i[0] for i in value], data_shape)

        for k, v in dms_results.items():
            logger.debug("DMS column %s rows: %d", k, len(v))
        dms_df = pd.DataFrame(dms_results).dropna(axis=1)
        result_columns = ['esm2', 'esmif', 'vaspa', 'poet', 'prosst']
        existing_columns = [col for col in result_columns if col in dms_df.columns]
        
        for col in existing_columns:
            dms_df[col] = dms_df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)


        if existing_columns:
            dms_df['average'] = dms_df[existing_columns].mean(axis=1)
            dms_df['rank'] = dms_df['average'].rank(ascending=False, method='min').astype(int)

        return dms_df
    
    def single_mask_recovery(self, seq, model_names: list, pdb=None):
        random.seed(self.seed) 
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Run single-position masking and collect reconstructed logits.
        results = {}
        
        for model_name in model_names:
            masked_seq = list(seq)
            reconstructed_logits = []
            mask_model = self.model_config[model_name]
            logger.info("Running single-mask recovery with %s", model_name)

            if 'esm2' in model_name or 'esm1v' in model_name:
                mask_tokenizer = self.esm2_tokenizer
                mask_token = mask_tokenizer.mask_token
            elif 'if' in model_name:
                mask_vocab = self.esmif_alphabet
                mask_token = np.inf

                coord, na_seq = esm.inverse_folding.util.load_coords(pdb, 'A')
                mask_tokenizer = CoordBatchConverter(mask_vocab)
                coord_, confidence, strs, tokens, padding_mask = mask_tokenizer([(coord, None, na_seq)])
                mask_coord_ = deepcopy(coord_)
                prev_output_tokens = tokens[:, :-1]
            else:
                raise ValueError(f'Invalid model name: {model_name}')
            
            with torch.no_grad():
                for i in tqdm(range(len(seq))):
                    if 'if' in model_name and pdb is not None:
                        mask_coord_[:,i] = mask_token
                        if_logits, extra = mask_model.forward(mask_coord_.to(self.device), 
                                                              padding_mask.to(self.device), 
                                                              confidence.to(self.device), 
                                                              prev_output_tokens.to(self.device))
                        
                        softmax_if = torch.nn.functional.softmax(if_logits.permute(0, 2, 1), dim=-1)
                        reconstructed_logits.append(softmax_if[0, i, 4:24])
                        mask_coord_[:, i] = coord_[:, i]
                    elif 'esm' in model_name:
                        masked_seq[i] = mask_token
                        masked_seq_str = ''.join(masked_seq)
                        esm_tk = mask_tokenizer(masked_seq_str, return_tensors="pt")
                        esm_tk = {k: v.to(self.device) for k, v in esm_tk.items()}
                        esm_ot = mask_model(**esm_tk)
                        esm_logits = F.softmax(esm_ot.logits, dim=-1)[:, 1:-1, 4:24]
                        reconstructed_logits.append(esm_logits[0, i, :])
                        masked_seq[i] = seq[i]

            results[model_name] = torch.stack(reconstructed_logits).cpu().detach().numpy()
        
        return results

    def run_single_mask(self, 
                        seq=None, 
                        pdb=None, 
                        out_name='single_mask_pos_result', 
                        model_list=['esm2']):
        # Run single-position mask recovery.
        if pdb is not None and seq is None:
            _, seq, _, _ = self.load_seqs(is_dms=False, pdb2dms=pdb)
        elif seq is None and pdb is None:
            raise ValueError("Sequence and PDB file cannot both be None.")

        results = self.single_mask_recovery(seq, model_list, pdb)

        return results
