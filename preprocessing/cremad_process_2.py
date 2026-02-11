#!/usr/bin/env python3
import os, argparse, numpy as np, torch, torchaudio, random
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, VideoMAEModel
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# ----------------------------
# CLI (match your original defaults)
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, default='/data/stonekab/crema-d-mirror/VideoFlash')
parser.add_argument('--audio_dir', type=str, default='/data/stonekab/crema-d-mirror/AudioWAV')
parser.add_argument('--save_path', type=str, default='/data/stonekab/cremad_feats_2.pt', help='single .pt file')
parser.add_argument('--frames', type=int, default=16, help='uniform frames per clip')
parser.add_argument('--batch_size', type=int, default=4, help='I/O loop batch')
parser.add_argument('--audio_sr', type=int, default=16000)

# NEW: HF cache dir to avoid filling ~/.cache
parser.add_argument('--hf_cache_dir', type=str, default='/data/stonekab/hf_cache')

parser.add_argument('--video_model', type=str, default='MCG-NJU/videomae-base-finetuned-kinetics')
parser.add_argument('--audio_model', type=str, default='microsoft/wavlm-base-plus')

# Optional seed
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
os.makedirs(args.hf_cache_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if device == "cuda":
    torch.cuda.manual_seed_all(args.seed)

# ----------------------------
# VIDEO: VideoMAE (same embedding style as UR-FUNNY script)
# ----------------------------
video_processor = AutoImageProcessor.from_pretrained(args.video_model, cache_dir=args.hf_cache_dir)
video_encoder   = VideoMAEModel.from_pretrained(args.video_model, cache_dir=args.hf_cache_dir).eval().to(device)

def sample_uniform_indices(T, K):
    if T <= 1:
        return np.zeros(K, dtype=int)
    return np.linspace(0, T - 1, K, endpoint=True).round().astype(int)

def load_k_frames(video_path, K=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    idx = sample_uniform_indices(len(vr), K)
    frames = vr.get_batch(idx).asnumpy()  # (K, H, W, 3), uint8
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Expected (K,H,W,3), got {frames.shape}")
    return frames

def encode_video_videomae(frames_np):
    """
    frames_np: (K,H,W,3) uint8
    returns: normalized embedding (D,) float16
    """
    inputs = video_processor(list(frames_np), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # typically (1, T, 3, H, W)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = video_encoder(pixel_values=pixel_values)
        else:
            out = video_encoder(pixel_values=pixel_values)

        feat = out.last_hidden_state.mean(dim=1)            # (1, D)
        feat = torch.nn.functional.normalize(feat, dim=-1)  # (1, D)
        return feat.squeeze(0).detach().cpu().numpy().astype("float16")

# ----------------------------
# AUDIO: WavLM-Base+
# ----------------------------
feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.audio_model, cache_dir=args.hf_cache_dir)
audio_encoder  = WavLMModel.from_pretrained(args.audio_model, cache_dir=args.hf_cache_dir).eval().to(device)

def load_audio_mono_resample(path, target_sr=16000):
    wav, sr = torchaudio.load(path)  # (C,N)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr

def parse_filename(fname):
    parts = os.path.splitext(fname)[0].split("_")
    # keep your original 3 fields, and (optionally) capture intensity if present
    subject_id = parts[0] if len(parts) > 0 else ""
    sentence_id = parts[1] if len(parts) > 1 else ""
    feeling = parts[2] if len(parts) > 2 else ""
    intensity = parts[3] if len(parts) > 3 else ""
    return subject_id, sentence_id, feeling, intensity

def encode_audio_wavlm(wav_1ch, sr):
    inputs = feat_extractor(wav_1ch.squeeze(0), sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = audio_encoder(**inputs)
        else:
            out = audio_encoder(**inputs)

        feat = out.last_hidden_state.mean(dim=1)            # (1, D)
        feat = torch.nn.functional.normalize(feat, dim=-1)  # (1, D)
        return feat.squeeze(0).detach().cpu().numpy().astype("float16")

# ----------------------------
# Main
# ----------------------------
video_files = sorted([f for f in os.listdir(args.video_dir) if f.lower().endswith(".flv")])
records = []

with torch.inference_mode():
    for base in tqdm(range(0, len(video_files), args.batch_size)):
        batch_files = video_files[base:base + args.batch_size]

        for fname in batch_files:
            subj_id, sent_id, feeling, intensity = parse_filename(fname)

            # ---- Video ----
            vpath = os.path.join(args.video_dir, fname)
            frames_np = load_k_frames(vpath, K=args.frames)
            vid_feat = encode_video_videomae(frames_np)  # (D,) float16

            # ---- Audio ----
            aname = os.path.splitext(fname)[0] + ".wav"
            apath = os.path.join(args.audio_dir, aname)
            wav, sr = load_audio_mono_resample(apath, target_sr=args.audio_sr)
            aud_feat = encode_audio_wavlm(wav, sr)  # (768,) float16

            records.append({
                "index": len(records),
                "file": fname,
                "subject_id": subj_id,
                "sentence_id": sent_id,
                "feeling": feeling,
                "intensity": intensity,
                "video_feat": vid_feat,
                "audio_feat": aud_feat
            })

        if device == 'cuda':
            torch.cuda.empty_cache()

torch.save(records, args.save_path)
print(f"Saved {len(records)} items to {args.save_path}")

