import os
import h5py
import torch
import numpy as np
from tqdm import tqdm

DATA_PATH = "/data/stonekab/mosi/mosi.hdf5"
SAVE_PATH = "/data/stonekab/mosi_feats.pt"
USE_FLOAT16 = True


def to_f16(x):
    arr = np.array(x, dtype=np.float32)
    return arr.astype(np.float16) if USE_FLOAT16 else arr.astype(np.float32)


def pad(arr, max_len):
    arr = np.array(arr)
    L, D = arr.shape
    if L >= max_len:
        return arr[:max_len]
    out = np.zeros((max_len, D), dtype=arr.dtype)
    out[:L] = arr
    return out


print(f"Loading MOSI from {DATA_PATH}")
f = h5py.File(DATA_PATH, "r")

import h5py

path = "/data/stonekab/mosi/mosi.hdf5"
with h5py.File(path, "r") as f:
    example = list(f.keys())[0]
    print("Example key:", example)
    print("Inside group:", list(f[example].keys()))

breakpoint()

# Dataset structure:
#   f["text"]["features"][id]
#   f["audio"]["features"][id]
#   f["vision"]["features"][id]
#   f["labels"][id]
#   f["splits"]["train"] → list of ids
#   f["splits"]["valid"]
#   f["splits"]["test"]

text_grp   = f["text"]["features"]
audio_grp  = f["audio"]["features"]
vision_grp = f["vision"]["features"]
labels_grp = f["labels"]["features"]
splits_grp = f["splits"]

# Collect all utterance IDs
utt_ids = list(text_grp.keys())

# Compute max lengths
print("Computing max lengths...")
max_text_len = max(text_grp[u].shape[0] for u in utt_ids)
max_audio_len = max(audio_grp[u].shape[0] for u in utt_ids)
max_vision_len = max(vision_grp[u].shape[0] for u in utt_ids)

print("Max lengths:", max_text_len, max_audio_len, max_vision_len)

# Create split lookup
split_map = {}
for u in splits_grp["train"][:]:
    split_map[u.decode("utf-8")] = "train"
for u in splits_grp["valid"][:]:
    split_map[u.decode("utf-8")] = "val"
for u in splits_grp["test"][:]:
    split_map[u.decode("utf-8")] = "test"

records = []

print("Processing samples...")
for utt_id in tqdm(utt_ids):
    # Features
    text = to_f16(text_grp[utt_id][:])
    audio = to_f16(audio_grp[utt_id][:])
    vision = to_f16(vision_grp[utt_id][:])

    text_len = text.shape[0]
    audio_len = audio.shape[0]
    vision_len = vision.shape[0]

    text_pad = pad(text, max_text_len)
    audio_pad = pad(audio, max_audio_len)
    vision_pad = pad(vision, max_vision_len)

    label = float(labels_grp[utt_id][0])  # scalar sentiment

    split = split_map.get(utt_id, "train")  # default to train if missing

    records.append({
        "utt_id": utt_id,
        "split": split,
        "label": label,
        "text_feat": text_pad,
        "text_len": text_len,
        "audio_feat": audio_pad,
        "audio_len": audio_len,
        "vision_feat": vision_pad,
        "vision_len": vision_len
    })

f.close()

print(f"Saving {len(records)} MOSI items → {SAVE_PATH}")
torch.save(records, SAVE_PATH)
print("Done.")
