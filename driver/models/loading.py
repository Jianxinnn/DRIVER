import torch
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter
import os
from driver.core.runtime import configure_runtime_environment

configure_runtime_environment()

import logging
from transformers import  EsmModel, EsmConfig, EsmForMaskedLM,EsmTokenizer
import pickle
from pathlib import Path
from Bio import SeqIO
import torch.nn.functional as F
import time

logger = logging.getLogger(__name__)

vaspa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
prosst_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
esm2_alphabet = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
esma_ab = ''.join(esm2_alphabet)
vaspa_to_index = {char: index for index, char in enumerate(vaspa_alphabet)}
index_map = [vaspa_to_index[char] for char in esm2_alphabet]
esm2_to_index = {char: index for index, char in enumerate(esm2_alphabet)}
index_to_esm2 = {index: char for char, index in esm2_to_index.items()}


def _resolve_hf_model_source(model_name):
    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_root is None:
        hf_home = os.environ.get("HF_HOME")
        cache_root = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    else:
        cache_root = Path(cache_root)

    repo_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.exists():
        return model_name

    ref_path = repo_cache / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text().strip()
        snapshot = snapshots_dir / revision
        if (snapshot / "config.json").exists():
            return str(snapshot)

    for snapshot in snapshots_dir.iterdir():
        if (snapshot / "config.json").exists():
            return str(snapshot)

    return model_name


def load_models(
        esm_model_name=None, #'facebook/esm2_t33_650M_UR50D',
        model_if1_name=None, #'esm_if1_gvp4_t16_142M_UR50',
        vaspag_model=False,
        poet=False,
        prosst=False,
        device = 'cuda',
        dropout=False
):
    # load model
    # esm2
    if esm_model_name is not None:
        logger.info("Loading %s", esm_model_name)
        if dropout:
            logger.info("Loading %s with dropout enabled", esm_model_name)
            config = EsmConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
            config.hidden_dropout_prob = 0.2
            # config.classifier_dropout = 0.1
            config.attention_probs_dropout_prob = 0.
            esm_model = EsmForMaskedLM.from_pretrained(esm_model_name, config=config)
            esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
            logger.debug("Loaded model: %s", esm_model)
        else:
            esm_model = EsmForMaskedLM.from_pretrained(esm_model_name)
            esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_name)


        return esm_model.to(device).eval(), esm_tokenizer 
        
    # esmif
    if model_if1_name is not None:
        logger.info("Loading ESM-IF model")
        model_if1, alphabet = esm.pretrained.load_model_and_alphabet(model_if1_name)
        if dropout:
            logger.debug("Loaded train-mode ESM-IF model: %s", model_if1.train())
            return model_if1.to(device).train(), alphabet
        return model_if1.to(device).eval(), alphabet
    
    # vaspag
    if vaspag_model:
        logger.info("Loading VespaG model")
        try:
            from vespag import get_vaspag_mdoel
            esm_model_3b, vaspag_model = get_vaspag_mdoel()
        except ImportError:
            from vespag.models.fnn import FNN

            vespag_home = Path(os.environ.get("VESPAG_HOME", "third_party/VespaG"))
            checkpoint = Path(os.environ.get("VESPAG_CHECKPOINT", vespag_home / "model_weights" / "state_dict_v2.pt"))
            if not checkpoint.exists():
                raise FileNotFoundError(
                    "VespaG checkpoint not found. Set VESPAG_CHECKPOINT, or set VESPAG_HOME "
                    "so $VESPAG_HOME/model_weights/state_dict_v2.pt exists."
                )
            esm_model_3b = EsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
            vaspag_model = FNN(hidden_layer_sizes=[256], input_dim=2560, output_dim=20, dropout_rate=0.2)
            vaspag_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        if dropout:
            return esm_model_3b.to(device).train(), vaspag_model.train().to(device)
        return esm_model_3b.to(device).eval(), vaspag_model.to(device)
    
    # poet
    if poet:
        logger.info("Loading PoET model")
        from driver.models.poet import get_poet_model
        poet = get_poet_model()
        if dropout:
            return poet.to(device).train()
        return  poet.to(device).eval()

    # ProSST
    if prosst:
        logger.info("Loading ProSST model")
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model_name = 'AI4Protein/ProSST-2048'
        model_source = _resolve_hf_model_source(model_name)
        prosst_tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        prosst_model = AutoModelForMaskedLM.from_pretrained(model_source, trust_remote_code=True).eval()
        return prosst_model.to(device), prosst_tokenizer



def esmif_logits_single(model_if1, 
                 alphabet,
                 pdb,
                 device='cuda'
                 ):
    
    coords, native_seq = esm.inverse_folding.util.load_coords(pdb, 'A')
    batch_converter = CoordBatchConverter(alphabet)

    batch = [(coords, None, native_seq)]
    coords_, confidence, strs, tokens, padding_mask = batch_converter(batch)
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]

    if_logits, extra = model_if1.forward(coords_.to(device), padding_mask.to(device), confidence.to(device), prev_output_tokens.to(device))
    softmax_if = torch.nn.functional.softmax(if_logits[0].permute(1,0), dim=-1)
    return native_seq, if_logits


def vaspag_logits_single(
        vaspag_model,
        esm2_3b,
        tokenizer,
        seq,
        device='cuda'
):
    esm_tk = tokenizer(seq, return_tensors="pt")
    esm_tk = {k: v.to(device) for k, v in esm_tk.items()}
    esm_ot = esm2_3b(**esm_tk, output_hidden_states=True)
    vaspa_g_logits = vaspag_model(esm_ot.hidden_states[-1])[0][1:-1, :].cpu().detach().numpy()
    assert vaspa_g_logits.shape[0] == len(seq)
    vaspa_g_logits = vaspa_g_logits[:, index_map]

    return vaspa_g_logits


def esm2_logits_single(
        esm_model,
        tokenizer,
        seq,
        device='cuda',
        dropout=False
):

    esm_tk = tokenizer(seq, return_tensors="pt")
    esm_tk = {k: v.to(device) for k, v in esm_tk.items()}
    esm_ot = esm_model(**esm_tk)
    esm_logits = F.softmax(esm_ot.logits[0], -1)[1:-1, 4:24]
    assert esm_logits.shape[0] == len(seq)
    return esm_logits

def esmif_logits_batch(model_if1, 
                       alphabet,
                       pdbs,
                       device='cuda'):
    
    batch_data = []
    for pdb in pdbs:
        coords, native_seq = esm.inverse_folding.util.load_coords(pdb, 'A')
        batch_data.append((coords, None, native_seq))
        
    batch_converter = CoordBatchConverter(alphabet)
    coords_, confidence, strs, tokens, padding_mask = batch_converter(batch_data)
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]

    if_logits, extra = model_if1.forward(coords_.to(device), padding_mask.to(device), confidence.to(device), prev_output_tokens.to(device))
    softmax_if = torch.nn.functional.softmax(if_logits.permute(1, 0, 2), dim=-1)
    return [strs, if_logits]

def vaspag_logits_batch(
        vaspag_model,
        esm2_3b,
        tokenizer,
        seqs,
        device='cuda'
):
    esm_tk = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
    esm_tk = {k: v.to(device) for k, v in esm_tk.items()}
    esm_ot = esm2_3b(**esm_tk, output_hidden_states=True)
    vaspa_g_logits = vaspag_model(esm_ot.hidden_states[-1])
    vaspa_g_logits = vaspa_g_logits[:, 1:-1, :].cpu().detach().numpy()

    # Adjust logits for each sequence
    batch_logits = []
    for seq, logits in zip(seqs, vaspa_g_logits):
        assert logits.shape[0] == len(seq)
        batch_logits.append(logits[:, index_map])

    return batch_logits

def esm2_logits_batch(
        esm_model,
        tokenizer,
        seqs,
        device='cuda',
        dropout=False
):
    esm_tk = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
    esm_tk = {k: v.to(device) for k, v in esm_tk.items()}
    esm_ot = esm_model(**esm_tk)
    esm_logits = F.softmax(esm_ot.logits[:, 1:-1, 4:24], dim=-1).cpu().detach().numpy()

    batch_logits = []
    for seq, logits in zip(seqs, esm_logits):
        assert logits.shape[0] == len(seq)
        batch_logits.append(logits)

    return batch_logits
