import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
import torch.nn.functional as F
import numpy as np
import pandas as pd
# from decord import VideoReader
# from decord import cpu
# import librosa
from src.multimodal_projector import MultiLoReFT
from src.utils import get_dino_preprocess
import wandb


class CremadDataset(Dataset):
    # def __init__(self, split='train', device='cuda', embedding_cache_dir="/data/stonekab/cached_cremad_feats"):
    #     cache_path = os.path.join(embedding_cache_dir, f"cached_cremad_feats.pt")
    def __init__(self, split='train', device='cuda', embedding_cache_dir="/data/stonekab/"):
        cache_path = os.path.join(embedding_cache_dir, f"cremad_feats_2.pt")
        self.device = device
        self.embedding_cache = torch.load(cache_path)
        
        # You might adjust this to point to your actual raw data folder
        self.video_dir = "/data/stonekab/CREMA-D/VideoFlash"
        self.audio_dir = "/data/stonekab/CREMA-D/AudioWAV"
        # Create a consistent split of IDs for train/validation/test
        all_ids = list(range(len(self.embedding_cache)))
        random.seed(42)  # Ensure reproducibility
        random.shuffle(all_ids)

        train_split = int(0.7 * len(all_ids))
        val_split = int(0.85 * len(all_ids))
        self.demographics_df = pd.read_csv('/data/stonekab/crema-d-mirror/VideoDemographics.csv')

        if split == 'train':
            self.ids = all_ids[:train_split]
            self.embedding_cache = self.embedding_cache[:train_split]
        elif split == 'val':
            self.ids = all_ids[train_split:val_split]
            self.embedding_cache = self.embedding_cache[train_split:val_split]
        elif split == 'test':
            self.ids = all_ids[val_split:]
            self.embedding_cache = self.embedding_cache[val_split:]
        else:
            raise ValueError(f"Unsupported split type: {split}")
        
        # Infer dims from first record
        v0 = self.embedding_cache[0]["video_feat"]
        t0 = self.embedding_cache[0]["audio_feat"]
        self.video_dim = int(len(v0))
        self.audio_dim = int(len(t0))

    def __len__(self):
        return len(self.embedding_cache)

    def __getitem__(self, idx):
        cached = self.embedding_cache[idx]
        # filename = self.filenames[idx]

        # Parse identifiers
        subject_id = cached["subject_id"]
        sentence_id = cached["sentence_id"]
        emotion = cached["feeling"]

        
        x1_placeholder = torch.zeros(1)  # You can also use torch.empty(0) if preferred
        x2_placeholder = ""
        # Find the row in the demographics data that matches the subject_id
        demographics_info = self.demographics_df[self.demographics_df['ActorID'] == int(subject_id)].iloc[0]

        sex_entries = ["Female", "Male"]
        race_entries = ["Caucasian", "African American", "Asian", "Hispanic", "Unknown"]
        ethnicity_entries = ["Hispanic", "Not Hispanic", "Unknown"]
        # Extract additional information
        age = demographics_info['Age']
        sex = sex_entries.index(demographics_info['Sex'])
        race = race_entries.index(demographics_info['Race'])
        ethnicity = ethnicity_entries.index(demographics_info['Ethnicity'])

        audio_feat = F.normalize(torch.Tensor(cached["audio_feat"]), dim=0)#[0]
        video_feat = F.normalize(torch.Tensor(cached["video_feat"]), dim=0)

        return video_feat, audio_feat, x1_placeholder, x2_placeholder, int(subject_id), sentence_id, emotion, age, sex, race, ethnicity

def main(lr, bs, rank, prune_th, seed_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_dataset = CremadDataset(split='train')
    val_dataset = CremadDataset(split='val')
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=True)

    projection_model = MultiLoReFT(
        input_dims=[train_dataset.video_dim, train_dataset.audio_dim],   # adjust if needed: video_feat dim, audio_feat dim
        shared_rank=rank,
        specific_rank=rank,
        pruning_threshold=prune_th,
        device=device,
        staging=False,
        pruning=True,
        shared_R_mode="pad",
        dataset_name="cremad"
    ).to(device)

    early_stopping_config = {
        "shared": {"patience": 20, "min_improvement_ratio": 0.001, "max_epochs": 300},
        "private": {"patience": 20, "min_improvement_ratio": 0.001, "max_epochs": 300},
        "joint": {"patience": 150, "min_improvement_ratio": 0.001, "max_epochs": 8000}
    }

    projection_model.train_projection(
        train_dataloader,
        val_dataloader,
        early_stopping_config,
        lr=lr,
        epochs=6000,
        exp_name='multi_loreft_lr%.4f_bs%d_rank%d_prune%.2f_%d_no_stage'%(lr, bs, rank, prune_th, seed_id),
        dataset_name="cremad",
        en_tokenizer=None,  # Not needed
        fr_tokenizer=None,
    )

if __name__ == "__main__":
    ranks = [700]
    for lr in [1e-3]:
        for bs in [256]:
            for rank in ranks:
                for prune_th in [0.1]:
                    run_name = 'cremad_multi_loreft_lr%.4f_bs%d_rank%d_prune%.2f'%(lr, bs, rank, prune_th)
                    wandb.init(project="MultiLoReFT", name=run_name, config={"lr": lr, "batch": bs, "rank": rank, "prune_th": prune_th})
                    for seed_id in range(3):
                        main(lr, bs, rank, prune_th, seed_id)
                    wandb.finish()
