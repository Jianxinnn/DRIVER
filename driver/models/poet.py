import sys
import os

poet_home = os.environ.get("POET_HOME")
if poet_home:
    sys.path.insert(0, poet_home)
import string
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

import logging

logger = logging.getLogger(__name__)


from poet.alphabets import Uniprot21
from poet.fasta import parse_stream
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET
from poet.msa.sampling import MSASampler, NeighborsSampler

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()
PBAR_POSITION = 1
T = TypeVar("T", np.ndarray, torch.Tensor)

def append_startstop(x: T, alphabet: Uniprot21) -> T:
    x_ndim = x.ndim
    assert x_ndim in {1, 2}
    if x_ndim == 1:
        x = x[None, :]

    if isinstance(x, torch.Tensor):
        empty_func = torch.empty
    else:
        empty_func = np.empty
    x_ = empty_func((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    x_[:, -1] = alphabet.stop_token
    x_[:, 1:-1] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_

def get_seqs_from_fastalike(filepath: Path) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> np.ndarray:
    # Remove lowercase insertions from A3M sequences.
    cleaned_sequences = [
        s.translate(None, delete=ASCII_LOWERCASE_BYTES) for s in msa_sequences
    ]
    # Find the most common sequence length.
    length_counts = {}
    for seq in cleaned_sequences:
        length = len(seq)
        length_counts[length] = length_counts.get(length, 0) + 1
    
    most_common_length = max(length_counts, key=length_counts.get)
    logger.info(f"Most common sequence length: {most_common_length}")
    
    # Keep sequences matching the most common length.
    filtered_sequences = [seq for seq in cleaned_sequences if len(seq) == most_common_length]
    
    logger.info(f"Original number of sequences: {len(cleaned_sequences)}")
    logger.info(f"Number of sequences after filtering: {len(filtered_sequences)}")
    
    # Encode and stack sequences.
    try:
        encoded_msa = np.vstack([alphabet.encode(seq) for seq in filtered_sequences])
        logger.info(f"Encoded MSA shape: {encoded_msa.shape}")
        return encoded_msa
    except ValueError as e:
        logger.error(f"Error during encoding and stacking: {e}")
        raise

def sample_msa_sequences(
    get_sequence_fn: Callable[[int], bytes],
    sample_idxs: Sequence[int],
    max_tokens: int,
    alphabet: Uniprot21,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    truncate: bool = True,
) -> list[np.ndarray]:
    assert alphabet.start_token != -1
    assert alphabet.stop_token != -1
    if not shuffle:
        assert shuffle_seed is None

    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break

    # shuffle order and truncate to max tokens
    if shuffle:
        rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random
        )
        final_permutation = rng.permutation(len(seqs))
    else:
        final_permutation = np.arange(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        if truncate and (total_tokens + len(seq) > max_tokens):
            seq = seq[: max_tokens - total_tokens]
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def jit_warmup(embedding_model: PoET, alphabet: Uniprot21, device='cpu'):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().to(device)
    segment_sizes = torch.tensor(segment_sizes).long().to(device)
    _ = embedding_model.embed(x.unsqueeze(0), segment_sizes.unsqueeze(0))



def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
    device='cuda'
) -> np.ndarray:
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory,
        batch_size=batch_size,
        length=max_variant_length - 1,  # discount stop token
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    embeds = []
    LOGITS = []
    if pbar_position is not None:
        pbar = trange(
            0,
            len(variants),
            batch_size,
            desc=f"[{pbar_position}] decoding",
            leave=False,
            position=pbar_position,
        )
    else:
        pbar = range(0, len(variants), batch_size)
    for start_idx in pbar:
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True,
            padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants,
                (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        assert (this_variants == alphabet.gap_token).sum() == 0
        this_variants = this_variants.to(device)
        logits, embed = model.logits(this_variants[:, :-1], memory, preallocated_memory=True,return_embeddings=True)
        targets = this_variants[:, 1:]
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.cpu().numpy())
        embeds.append(embed[:,1:,:].cpu().numpy())
        LOGITS.append(logits[:,1:,:].cpu().numpy())

    return np.hstack(logps), np.vstack(embeds), np.vstack(LOGITS)

def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
    device='cuda'
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).to(device)
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).to(device)
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
        device=device
    )


alphabet = Uniprot21(
        include_gap=True, include_startstop=True, distinct_startstop=True
    )

def get_poet_model(device='cpu'):
    ckpt = os.environ.get("POET_CHECKPOINT")
    if ckpt is None:
        default_home = Path(os.environ.get("POET_HOME", "third_party/PoET"))
        ckpt = str(default_home / "data" / "poet.ckpt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            "PoET checkpoint not found. Set POET_CHECKPOINT, or set POET_HOME "
            "so $POET_HOME/data/poet.ckpt exists."
        )
    load_ckpt = torch.load(ckpt, map_location='cpu')
    model = PoET(**load_ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict(
            {k.split(".", 1)[1]: v for k, v in load_ckpt["state_dict"].items()}
        )
    model = model.to(device)
    model.half().eval()
    # total_params = sum(p.numel() for p in model.parameters())
    # jit_warmup(model, alphabet, device)
    return model


def get_poet_predict(model, msa_sequences, variants, msa, device='cuda', \
                     max_tokens=122880, max_similarity=0.95, batch_size=2, seed=2024, bi_direct=True, if_embeds=False):
    global alphabet
    logps = []

    with torch.no_grad():
        sampler = MSASampler(
            method=NeighborsSampler(
                can_use_torch=False,
            ),
            max_similarity=max_similarity,
        )

        sample_idxs = sampler.get_sample_idxs(
            msa=msa,
            gap_token=alphabet.gap_token,
            seed=seed,
        )
        this_msa_sequences = sample_msa_sequences(
            get_sequence_fn=lambda ii: msa_sequences[ii]
            .upper()
            .translate(None, delete=b"-"),
            sample_idxs=sample_idxs,
            max_tokens=max_tokens,
            alphabet=alphabet,
            shuffle_seed=seed,
            truncate=False,
        )
        forward_logps, forward_embeds, forward_logits = get_logps_tiered_fast(
            msa_sequences=this_msa_sequences,
            variants=variants,
            model=model,
            batch_size=batch_size,
            alphabet=alphabet,
            pbar_position=PBAR_POSITION,
            device=device
        )

        if bi_direct:
            backward_logps, backword_embeds, backward_logits = get_logps_tiered_fast(
                msa_sequences=[np.ascontiguousarray(s[::-1]) for s in this_msa_sequences],
                variants=[np.ascontiguousarray(s[::-1]) for s in variants],
                model=model,
                batch_size=batch_size,
                alphabet=alphabet,
                pbar_position=PBAR_POSITION,
                device=device
            )
            this_logps = (forward_logps + backward_logps) / 2
        else:
            this_logps = forward_logps

        logps.append(this_logps)
    if if_embeds:
        return logps[0], forward_embeds, forward_logits
    else:
        return logps[0]

def set_dropout(model, dropout_probability=0.1):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_probability
