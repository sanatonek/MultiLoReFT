#!/usr/bin/env python3
import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import random
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb

from src.multimodal_projector import MultiLoReFT


class UrFunnyDataset(Dataset):
    """
    Expects a torch-saved list[dict] where each record has at least:
      - split: 'train' | 'dev' | 'test'
      - id: segment id (string)
      - label: 0/1 humor label
      - video_feat: np.ndarray or list (float16/float32) OR None
      - text_feat:  np.ndarray or list (float16/float32) OR None
      - punchline_text/context_text optional

    This matches the output schema of the UR-FUNNY extraction script we discussed.
    """

    def __init__(
        self,
        split: str = "train",                 # 'train' | 'val' | 'test'
        device: str = "cuda",
        raw_video: bool = False,
        embedding_cache_dir: str = "/data/stonekab/",
        cache_name: str = "urfunny_feats.pt",
        require_video: bool = True,
        require_text: bool = True,
    ):
        super().__init__()
        self.device = device
        self.raw_video = raw_video

        cache_path = os.path.join(embedding_cache_dir, cache_name)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"UR-FUNNY cache not found: {cache_path}")

        all_records = torch.load(cache_path, map_location="cpu")

        split_map = {"train": "train", "val": "dev", "test": "test"}
        if split not in split_map:
            raise ValueError(f"Unsupported split={split}. Use train|val|test.")
        target_split = split_map[split]

        # Filter by split and optionally require both modalities
        recs = []
        for r in all_records:
            if r.get("split") != target_split:
                continue
            if self.raw_video:
                if require_video and (r.get("video_feat_videomae") is None):
                    continue
            else:
                if require_video and (r.get("visual_openface_full") is None):
                    continue
            if require_text and (r.get("text_feat") is None):
                continue
            recs.append(r)

        if len(recs) == 0:
            raise RuntimeError(
                f"No usable UR-FUNNY records after filtering. "
                f"split={target_split}, require_video={require_video}, require_text={require_text}. "
                f"Check your extraction output + video availability."
            )

        self.records = recs
        # Infer dims from first record
        if self.raw_video:
            v0 = self.records[0]["video_feat_videomae"]
        else:
            v0 = self.records[0]["visual_openface_full"]
        t0 = self.records[0]["text_feat"]
        self.video_dim = int(len(v0))
        self.text_dim = int(len(t0))

        print(
            f"[UrFunnyDataset] split={split} ({target_split}) "
            f"n={len(self.records)} video_dim={self.video_dim} text_dim={self.text_dim}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        # Modalities
        if self.raw_video:
            video_feat = F.normalize(torch.tensor(r["video_feat_videomae"], dtype=torch.float32), dim=0)
        else:
            video_feat = F.normalize(torch.tensor(r["visual_openface_full"], dtype=torch.float32), dim=0)
        # video_feat = F.normalize(torch.tensor(r["video_feat_videomae"], dtype=torch.float32), dim=0)
        text_feat  = F.normalize(torch.tensor(r["text_feat"],  dtype=torch.float32), dim=0)

        # Placeholders to keep signature similar to your CREMA-D dataset
        x1_placeholder = torch.zeros(1)
        x2_placeholder = ""

        # Metadata
        seg_id = r.get("id", "")
        y = int(r.get("label", 0))

        punch = r.get("punchline_text", "")
        ctx = r.get("context_text", "")

        # For compatibility with CREMA-D return structure (extra fields):
        # subject_id, sentence_id, emotion, age, sex, race, ethnicity
        # UR-FUNNY doesn’t have these → fill with safe placeholders.
        subject_id = 0
        sentence_id = seg_id
        emotion = y         

        return (
            video_feat, text_feat,
            x1_placeholder, x2_placeholder, emotion
        )


def main(lr, bs, rank, prune_th, seed_id,
         embedding_cache_dir="/data/stonekab/",
         cache_name="urfunny_feats.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Reproducibility
    random.seed(42 + seed_id)
    torch.manual_seed(42 + seed_id)
    if device == "cuda":
        torch.cuda.manual_seed_all(42 + seed_id)

    use_raw_video = True
    train_dataset = UrFunnyDataset(split="train", device=device, raw_video=use_raw_video,
                                   embedding_cache_dir=embedding_cache_dir,
                                   cache_name=cache_name,
                                   require_video=True, require_text=True)
    val_dataset   = UrFunnyDataset(split="val", device=device,
                                   raw_video=use_raw_video,
                                   embedding_cache_dir=embedding_cache_dir,
                                   cache_name=cache_name,
                                   require_video=True, require_text=True)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False)
    val_dataloader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False, drop_last=False)

    # Infer dims from dataset (don’t hardcode 768/1024)
    in_dims = [train_dataset.video_dim, train_dataset.text_dim]

    projection_model = MultiLoReFT(
        input_dims=in_dims,
        shared_rank=rank,
        specific_rank=rank,
        pruning_threshold=prune_th,
        device=device,
        staging=False,
        pruning=True,
        shared_R_mode="pad",
        dataset_name="urfunny",
    ).to(device)

    early_stopping_config = {
        "shared":  {"patience": 20,  "min_improvement_ratio": 0.001, "max_epochs": 300},
        "private": {"patience": 20,  "min_improvement_ratio": 0.001, "max_epochs": 300},
        "joint":   {"patience": 150, "min_improvement_ratio": 0.001, "max_epochs": 1800},
    }

    projection_model.train_projection(
        train_dataloader,
        val_dataloader,
        early_stopping_config,
        lr=lr,
        epochs=10000,
        exp_name=f"multi_loreft_lr{lr:.4f}_bs{bs}_rank{rank}_prune{prune_th:.2f}_{seed_id}_no_stage",
        dataset_name="urfunny",
        en_tokenizer=None,
        fr_tokenizer=None,
    )


if __name__ == "__main__":
    # If you prefer CLI for cache path/name:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_cache_dir", type=str, default="/data/stonekab/")
    parser.add_argument("--cache_name", type=str, default="urfunny_feats.pt")
    args, _ = parser.parse_known_args()

    ranks = [700]
    for lr in [1e-3]:
        for bs in [64]:
            for rank in ranks:
                for prune_th in [0.1]:
                    run_name = f"urfunny_multi_loreft_lr{lr:.4f}_bs{bs}_rank{rank}_prune{prune_th:.2f}"
                    wandb.init(project="MultiLoReFT", name=run_name,
                               config={"lr": lr, "batch": bs, "rank": rank, "prune_th": prune_th})
                    for seed_id in range(3):
                        main(lr, bs, rank, prune_th, seed_id,
                             embedding_cache_dir=args.embedding_cache_dir,
                             cache_name=args.cache_name)
                    wandb.finish()

