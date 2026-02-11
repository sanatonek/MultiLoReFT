#!/usr/bin/env python3
"""
Build a binary (yes/no) VQA dataset tensor from HuggingFaceM4/VQAv2 in ONE .pt file.

- Loads split from HuggingFaceM4/VQAv2
- Filters to yes/no questions (VQAv2-style multi-annotator answers)
- Encodes images + questions with selected backbones
- Saves a single .pt payload with aligned features and labels

Example:
python vqa_yesno_build_single_pt.py \
  --split train --out /cache/vqav2_yesno_train.pt \
  --vision_backbone openclip_vith14 --text_backbone openclip \
  --batch_size 64 --fp16
"""

import argparse, sys
from collections import Counter
from typing import List
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

YES_TOKENS = {"yes", "yeah", "yep", "yea", "affirmative", "correct", "true"}
NO_TOKENS  = {"no", "nope", "nah", "negative", "incorrect", "false"}

# ---------------- Args ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="HuggingFaceM4/VQAv2")
    p.add_argument("--split", type=str, default="train", help="train | validation | test")
    p.add_argument("--cache_dir", type=str, default="/data/stonekab")
    p.add_argument("--out", type=str, default="/data/stonekab/vqa_yesno_train.pt")

    p.add_argument("--vision_backbone", type=str, default="openclip_vith14",
                   choices=["openclip_vith14", "clip_vitl14_336", "siglip_384", "dinov2_g"])
    p.add_argument("--text_backbone", type=str, default="openclip",
                   choices=["openclip", "siglip"])

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tie_no", action="store_true", help="If yes/no votes tie, map to NO (default: YES).")
    p.add_argument("--max_examples", type=int, default=None, help="Optional cap for debugging.")
    return p.parse_args()

# ------------- Label helpers -------------
def _norm(a: str) -> str:
    return a.strip().lower()

def vqav2_yesno_label(answers: List[str], tie_no: bool) -> int:
    """Return 0/1 for no/yes, or raise ValueError if not yes/no."""
    c = Counter(_norm(a) for a in answers if a)
    yes_votes = sum(c.get(tok, 0) for tok in YES_TOKENS)
    no_votes  = sum(c.get(tok, 0) for tok in NO_TOKENS)
    if yes_votes == 0 and no_votes == 0:
        raise ValueError("not_yesno")
    if yes_votes > no_votes: return 1
    if no_votes  > yes_votes: return 0
    return 0 if tie_no else 1

# -------- Vision encoders (PIL -> tensor -> feats) --------
def build_vision(vision_backbone: str, device: str, fp16: bool):
    if vision_backbone in ["openclip_vith14", "clip_vitl14_336"]:
        import open_clip
        if vision_backbone == "openclip_vith14":
            model_name, ckpt = "ViT-H-14", "laion2b_s32b_b79k"
        else:
            model_name, ckpt = "ViT-L-14-336", "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt, device=device)
        vision = model.visual.eval()
        feat_dim = int(vision.output_dim)
        if fp16: vision.half()

        @torch.inference_mode()
        def encode_pils(pils: List[Image.Image]) -> torch.Tensor:
            ims = [preprocess(img.convert("RGB")) for img in pils]
            batch = torch.stack(ims, 0).to(device, non_blocking=True)
            if fp16: batch = batch.half()
            feats = vision(batch)                           # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu()
        name = vision_backbone

    elif vision_backbone == "siglip_384":
        from transformers import AutoImageProcessor, SiglipVisionModel
        ckpt = "google/siglip-so400m-patch14-384"
        processor = AutoImageProcessor.from_pretrained(ckpt)
        model = SiglipVisionModel.from_pretrained(ckpt).to(device).eval()
        if fp16: model.half()
        feat_dim = int(model.config.hidden_size)

        @torch.inference_mode()
        def encode_pils(pils: List[Image.Image]) -> torch.Tensor:
            inputs = processor(images=[p.convert("RGB") for p in pils], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            if fp16: pixel_values = pixel_values.half()
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output                     # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu()
        name = "siglip_384"

    elif vision_backbone == "dinov2_g":
        import timm
        model = timm.create_model("vit_giant_patch14_224.dino", pretrained=True)
        model.reset_classifier(0)
        model = model.to(device).eval()
        if fp16: model.half()
        feat_dim = int(model.num_features)
        data_cfg = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)

        @torch.inference_mode()
        def encode_pils(pils: List[Image.Image]) -> torch.Tensor:
            ims = [preprocess(img.convert("RGB")) for img in pils]
            batch = torch.stack(ims, 0).to(device, non_blocking=True)
            if fp16: batch = batch.half()
            feats = model.forward_features(batch)
            if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                feats = feats["x_norm_clstoken"]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu()
        name = "dinov2_g"

    else:
        raise ValueError(f"Unknown vision_backbone: {vision_backbone}")

    return encode_pils, feat_dim, name

# -------- Text encoders (string -> feats) --------
def build_text(text_backbone: str, device: str, fp16: bool, pair_with_vision: str):
    if text_backbone == "openclip":
        import open_clip
        # Pair tokenizer/tower with vision for best alignment
        if pair_with_vision == "clip_vitl14_336":
            model_name, ckpt = "ViT-L-14-336", "openai"
        else:
            model_name, ckpt = "ViT-H-14", "laion2b_s32b_b79k"
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=ckpt, device=device)
        if fp16: model.half()
        tokenizer = open_clip.get_tokenizer(model_name)
        text_dim = int(model.text_projection.shape[-1])

        @torch.inference_mode()
        def encode_text(strs: List[str]) -> torch.Tensor:
            toks = tokenizer(strs).to(device)
            feats = model.encode_text(toks)                # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu()
        name = "openclip_text"

    elif text_backbone == "siglip":
        from transformers import AutoTokenizer, SiglipTextModel
        ckpt = "google/siglip-so400m-patch14-384"
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = SiglipTextModel.from_pretrained(ckpt).to(device).eval()
        if fp16: model.half()
        text_dim = int(model.config.hidden_size)

        @torch.inference_mode()
        def encode_text(strs: List[str]) -> torch.Tensor:
            enc = tok(strs, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc)
            feats = out.pooler_output                     # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu()
        name = "siglip_text"

    else:
        raise ValueError(f"Unknown text_backbone: {text_backbone}")

    return encode_text, text_dim, name

# ---------------- Main ----------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] split={args.split} device={device} vision={args.vision_backbone} text={args.text_backbone} fp16={args.fp16}")

    # 1) Load HF dataset split
    ds = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)

    # Expected fields: 'image' (PIL), 'question' (str), 'answers' (dict with 'text' list), ids
    # Probe a row to assert format
    ex0 = ds[0]
    if "image" not in ex0 or "question" not in ex0 or "answers" not in ex0:
        print("Unexpected dataset schema. Need fields: image, question, answers", file=sys.stderr)
        sys.exit(1)

    # 2) Filter to yes/no and collect raw fields
    keep_imgs, keep_qs, keep_labels, keep_qids, keep_imgids = [], [], [], [], []
    skipped_non_yesno = 0

    for i, ex in enumerate(tqdm(ds, desc="filter_yesno", ncols=80)):
        # answers is usually a dict {'text': [...], 'answer_start': [...] } — we only use 'text'
        answers = ex.get("answers", {})
        texts = answers.get("text", answers) if isinstance(answers, dict) else answers
        try:
            label = vqav2_yesno_label(texts, tie_no=args.tie_no)
        except Exception:
            skipped_non_yesno += 1
            continue

        img = ex["image"]                              # PIL.Image.Image
        if not isinstance(img, Image.Image):
            # datasets sometimes stores as dict with 'path'
            # but HF VQAv2 should be PIL; still, be defensive:
            try:
                img = Image.open(ex["image"]["path"]).convert("RGB")
            except Exception:
                skipped_non_yesno += 1
                continue

        keep_imgs.append(img.convert("RGB"))
        keep_qs.append(ex["question"])
        keep_labels.append(label)
        keep_qids.append(int(ex.get("question_id", ex.get("id", i))))
        keep_imgids.append(int(ex.get("image_id", -1)))

        if args.max_examples and len(keep_labels) >= args.max_examples:
            break

    if not keep_labels:
        print("No yes/no examples found.", file=sys.stderr)
        sys.exit(1)

    print(f"[filter] kept={len(keep_labels)}  skipped_non_yesno={skipped_non_yesno}")

    # 3) Build encoders
    encode_pils, vdim, vname = build_vision(args.vision_backbone, device, args.fp16)
    encode_text, tdim, tname = build_text(args.text_backbone, device, args.fp16, pair_with_vision=args.vision_backbone)

    # 4) Encode features in batches
    B = max(1, args.batch_size)

    # Images
    img_feats_chunks = []
    for i in tqdm(range(0, len(keep_imgs), B), desc="encode_images", ncols=80):
        chunk = keep_imgs[i:i+B]
        feats = encode_pils(chunk)          # (b, vdim) CPU float32
        img_feats_chunks.append(feats)
        if device == "cuda": torch.cuda.empty_cache()
    image_feats = torch.cat(img_feats_chunks, dim=0)   # (N, vdim)

    # Questions
    q_feats_chunks = []
    for i in tqdm(range(0, len(keep_qs), B), desc="encode_questions", ncols=80):
        chunk = keep_qs[i:i+B]
        feats = encode_text(chunk)          # (b, tdim) CPU float32
        q_feats_chunks.append(feats)
        if device == "cuda": torch.cuda.empty_cache()
    question_feats = torch.cat(q_feats_chunks, dim=0)  # (N, tdim)

    labels = torch.tensor(keep_labels, dtype=torch.long)  # 0=no, 1=yes

    # 5) Save single .pt
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "dataset": args.dataset_name,
            "split": args.split,
            "vision_backbone": vname, "vision_dim": int(vdim),
            "text_backbone": tname,   "text_dim": int(tdim),
            "num_examples": int(len(labels)),
        },
        "qids": keep_qids,            # List[int]
        "image_ids": keep_imgids,     # List[int]
        "questions": keep_qs,         # List[str]
        "image_feats": image_feats,   # (N, vdim) float32 (use --fp16-save if desired)
        "question_feats": question_feats,  # (N, tdim)
        "labels": labels,             # (N,)
    }

    # Optional smaller file: cast to fp16 on disk by toggling here if you want
    # (We keep the command-line flag only for run-time compute; storage can be changed here.)
    # If you *do* want to store fp16, uncomment:
    # payload["image_feats"] = payload["image_feats"].to(torch.float16)
    # payload["question_feats"] = payload["question_feats"].to(torch.float16)

    torch.save(payload, out_path)
    print(f"[saved] {out_path}")
    print(f" image_feats:    {tuple(image_feats.shape)}")
    print(f" question_feats: {tuple(question_feats.shape)}")
    print(f" labels:         {tuple(labels.shape)}")

if __name__ == "__main__":
    main()
