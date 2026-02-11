#!/usr/bin/env python3
import os, argparse, pickle, glob, random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, VideoMAEModel


# ----------------------------
# Pickle utils (UR-FUNNY SDK)
# ----------------------------
def load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def norm_id(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            x = x.item()
    except Exception:
        pass
    return str(x)

def norm_inner_keys(obj):
    # Some SDK pickles use bytes keys inside dicts
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k
            out[kk] = norm_inner_keys(v)
        return out
    if isinstance(obj, list):
        return [norm_inner_keys(x) for x in obj]
    return obj

def words_to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        if isinstance(x[0], (list, tuple)):
            return " ".join([" ".join(map(str, s)) for s in x])
        return " ".join(map(str, x))
    return str(x)


# ----------------------------
# Video utils
# ----------------------------
def sample_uniform_indices(T: int, K: int) -> np.ndarray:
    if T <= 1:
        return np.zeros(K, dtype=np.int64)
    return np.linspace(0, T - 1, K, endpoint=True).round().astype(np.int64)

def load_k_frames(video_path: str, K: int) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    idx = sample_uniform_indices(len(vr), K)
    frames = vr.get_batch(idx).asnumpy()  # (K,H,W,3) uint8
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Bad frames shape {frames.shape} for {video_path}")
    return frames

def build_video_index(video_root: str) -> Dict[str, str]:
    # recursive: handles subfolders
    idx = {}
    for p in glob.glob(os.path.join(video_root, "**", "*.*"), recursive=True):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in idx:
            idx[stem] = p
    return idx


# ----------------------------
# OpenFace pooling (FULL = context + punchline)
# ----------------------------
def _to_2d_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(np.float32)[None, :]
    if arr.ndim == 2:
        return arr.astype(np.float32)
    try:
        arr = np.array(x, dtype=np.float32)
        if arr.ndim == 2:
            return arr
    except Exception:
        pass
    return None

def pool_openface_full(openface_dict: Dict[str, Any], seg_id: str, pool: str = "mean") -> Optional[np.ndarray]:
    """
    openface_features_sdk.pkl entry typically has:
      - context_features: list of arrays [T_i, D]
      - punchline_features: array [T_p, D]
    We concatenate all context + punchline timesteps and pool -> 1D vector.
    """
    item = openface_dict.get(seg_id, None)
    if item is None:
        return None
    if isinstance(item, dict):
        item = norm_inner_keys(item)

    ctx_list = item.get("context_features", None)
    pun = item.get("punchline_features", None)

    ctx_arrays = []
    if isinstance(ctx_list, list):
        for s in ctx_list:
            a = _to_2d_array(s)
            if a is not None and a.size > 0:
                ctx_arrays.append(a)

    pun_a = _to_2d_array(pun)

    full_cat = None
    if len(ctx_arrays) > 0 and pun_a is not None:
        full_cat = np.concatenate([np.concatenate(ctx_arrays, axis=0), pun_a], axis=0)
    elif len(ctx_arrays) > 0:
        full_cat = np.concatenate(ctx_arrays, axis=0)
    elif pun_a is not None:
        full_cat = pun_a

    if full_cat is None or full_cat.size == 0:
        return None

    if pool == "last":
        v = full_cat[-1]
    else:
        v = full_cat.mean(axis=0)
    return v.astype(np.float16)


# ----------------------------
# Text packing: guarantee punchline + keep tail of context
# ----------------------------
def build_pair_input_ids(tokenizer, context: str, punchline: str,
                         max_len: int, max_punch_tokens: int) -> Tuple[List[int], List[int]]:
    """
    Build: [CLS] context_tail [SEP] punchline [SEP]
    Punchline is guaranteed to fit (up to max_punch_tokens).
    Context uses remaining budget, taking the tail (closest to punchline).
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    pun_ids = tokenizer.encode(punchline, add_special_tokens=False)

    if len(pun_ids) > max_punch_tokens:
        pun_ids = pun_ids[:max_punch_tokens]

    # [CLS] A [SEP] B [SEP] => 3 special tokens total
    ctx_budget = max_len - (len(pun_ids) + 3)
    if ctx_budget < 0:
        pun_ids = pun_ids[: max(1, max_len - 3)]
        ctx_ids = []
        ctx_budget = 0

    if len(ctx_ids) > ctx_budget:
        ctx_ids = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []

    input_ids = tokenizer.build_inputs_with_special_tokens(ctx_ids, pun_ids)
    attn_mask = [1] * len(input_ids)
    return input_ids, attn_mask


def main():
    parser = argparse.ArgumentParser()

    # UR-FUNNY inputs
    parser.add_argument("--features_dir", type=str, default="/data/stonekab/urfunny/")
    parser.add_argument("--video_dir", type=str, default="/data/stonekab/urfunny/videos/")

    # output
    parser.add_argument("--save_path", type=str, default="/data/stonekab/urfunny_feats.pt")

    # cache/temp dirs to avoid filling local_home
    parser.add_argument("--hf_cache_dir", type=str, default="/data/stonekab/hf_cache")
    parser.add_argument("--tmp_dir", type=str, default="/data/stonekab/hf_cache/tmp")

    # encoders
    parser.add_argument("--videomae_model", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--text_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--no_text_encoder", action="store_true", help="skip transformer text embeddings")

    # extraction settings
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--text_max_len", type=int, default=512)
    parser.add_argument("--max_punch_tokens", type=int, default=128)
    parser.add_argument("--text_batch_size", type=int, default=64)

    parser.add_argument("--openface_pool", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--only_split", type=str, default="all", choices=["all", "train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.hf_cache_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    # Route HF cache + temp downloads to big disk
    os.environ["HF_HOME"] = args.hf_cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(args.hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache_dir, "transformers")
    os.environ["TMPDIR"] = args.tmp_dir

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- load UR-FUNNY pickles ----
    folds_path   = os.path.join(args.features_dir, "data_folds.pkl")
    lang_path    = os.path.join(args.features_dir, "language_sdk.pkl")
    label_path   = os.path.join(args.features_dir, "humor_label_sdk.pkl")
    openface_path = os.path.join(args.features_dir, "openface_features_sdk.pkl")

    data_folds = load_pickle(folds_path)
    language   = load_pickle(lang_path)
    labels     = load_pickle(label_path)
    openface   = load_pickle(openface_path)

    # normalize keys
    language = {norm_id(k): norm_inner_keys(v) for k, v in language.items()}
    labels   = {norm_id(k): int(v) for k, v in labels.items()}
    openface = {norm_id(k): v for k, v in openface.items()}

    for sp in ["train", "dev", "test"]:
        if sp in data_folds:
            data_folds[sp] = [norm_id(x) for x in data_folds[sp]]

    # build id->split and ordered id list (official folds order)
    id2split: Dict[str, str] = {}
    for sp in ["train", "dev", "test"]:
        for seg_id in data_folds.get(sp, []):
            id2split[seg_id] = sp

    all_ids: List[str] = []
    for sp in ["train", "dev", "test"]:
        for seg_id in data_folds.get(sp, []):
            if seg_id in labels and seg_id in language:
                all_ids.append(seg_id)

    if args.only_split != "all":
        all_ids = [i for i in all_ids if id2split.get(i) == args.only_split]

    if args.limit and args.limit > 0:
        all_ids = all_ids[:args.limit]

    print(f"[UR-FUNNY] usable segments: {len(all_ids)} (split={args.only_split})")

    # ---- index videos recursively ----
    video_index = build_video_index(args.video_dir)
    print(f"[UR-FUNNY] video files indexed: {len(video_index)}")

    # ---- load models ----
    # VideoMAE
    videomae_proc = AutoImageProcessor.from_pretrained(args.videomae_model, cache_dir=args.hf_cache_dir)
    videomae_enc  = VideoMAEModel.from_pretrained(args.videomae_model, cache_dir=args.hf_cache_dir).eval().to(device)

    # Text encoder (optional)
    tok = enc = None
    if not args.no_text_encoder:
        tok = AutoTokenizer.from_pretrained(args.text_model, cache_dir=args.hf_cache_dir)
        enc = AutoModel.from_pretrained(args.text_model, cache_dir=args.hf_cache_dir).eval().to(device)

    # ---- prep text fields ----
    packed: List[Tuple[str, str, str, str, int]] = []
    for seg_id in all_ids:
        item = language[seg_id]
        ctx   = words_to_text(item.get("context_sentences", ""))
        punch = words_to_text(item.get("punchline_sentence", ""))
        y = int(labels[seg_id])
        sp = id2split.get(seg_id, "unknown")
        packed.append((seg_id, sp, ctx, punch, y))

    # ---- embed text in batches (punchline guaranteed) ----
    text_feats = None
    if not args.no_text_encoder:
        out_chunks = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(packed), args.text_batch_size), desc="Text"):
                batch = packed[i:i+args.text_batch_size]
                ids_list = []
                for (_, _, ctx, punch, _) in batch:
                    input_ids, attn = build_pair_input_ids(
                        tok, ctx, punch,
                        max_len=args.text_max_len,
                        max_punch_tokens=args.max_punch_tokens
                    )
                    ids_list.append({"input_ids": input_ids, "attention_mask": attn})

                padded = tok.pad(ids_list, padding=True, return_tensors="pt")
                padded = {k: v.to(device) for k, v in padded.items()}

                out = enc(**padded)
                emb = out.last_hidden_state[:, 0, :]             # CLS
                emb = torch.nn.functional.normalize(emb, dim=-1) # unit norm
                out_chunks.append(emb.detach().cpu().numpy().astype(np.float16))

        text_feats = np.concatenate(out_chunks, axis=0)

    # ---- video embedding ----
    def embed_videomae(video_path: str) -> np.ndarray:
        frames = load_k_frames(video_path, K=args.frames)  # (K,H,W,3)
        inputs = videomae_proc(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    vout = videomae_enc(pixel_values=pixel_values)
            else:
                vout = videomae_enc(pixel_values=pixel_values)

            feat = vout.last_hidden_state.mean(dim=1)  # mean pool tokens
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float16)

    # ---- assemble records ----
    records: List[Dict[str, Any]] = []
    matched_files = 0
    embedded_videomae = 0

    for i, (seg_id, sp, ctx, punch, y) in enumerate(tqdm(packed, desc="Video+OpenFace")):
        # OpenFace visual feature (always from pickle; may still be missing for some ids)
        visual_openface_full = pool_openface_full(openface, seg_id, pool=args.openface_pool)

        # Raw video embedding (optional; None if missing/failed)
        vpath = video_index.get(seg_id, "")
        video_feat_videomae = None

        if vpath and os.path.exists(vpath):
            matched_files += 1
            try:
                video_feat_videomae = embed_videomae(vpath)
                embedded_videomae += 1
            except Exception as e:
                print(f"[WARN] VideoMAE failed for {seg_id} ({vpath}): {e}")
                video_feat_videomae = None
        else:
            vpath = ""  # keep it clean if not found

        rec: Dict[str, Any] = {
            "index": i,
            "id": seg_id,
            "split": sp,
            "label": int(y),

            "video_file": vpath,
            "context_text": ctx,
            "punchline_text": punch,

            # Raw-video embedding (or None)
            "video_feat_videomae": video_feat_videomae,

            # Dataset-provided visual (OpenFace) pooled over context+punchline
            "visual_openface_full": visual_openface_full,
        }

        if text_feats is not None:
            rec["text_feat"] = text_feats[i]

        records.append(rec)

        if device == "cuda" and (i % 100 == 0):
            torch.cuda.empty_cache()

    torch.save(records, args.save_path)
    print(f"Saved {len(records)} items to {args.save_path}")
    print(f"Video files matched (exists): {matched_files}/{len(records)}")
    print(f"VideoMAE embeddings computed:  {embedded_videomae}/{len(records)}")


if __name__ == "__main__":
    main()
