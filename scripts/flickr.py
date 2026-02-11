import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import os
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils import get_dino_preprocess
from src.multimodal_projector import MultiLoReFT
from transformers import BertTokenizer, AutoTokenizer
import torch.nn.functional as F
import timm
import sys
import wandb



class Multi30KMixedLangDataset(Dataset):
    def __init__(self, split='train', device='cuda', embedding_cache_dir="/data/stonekab/cached_flickr_feats", return_raw=False):
        # self.dataset = load_dataset("romrawinjp/multi30k", split=split)
        self.dataset = load_dataset(
                    "romrawinjp/multi30k",
                    split=split,
                    cache_dir="/data/stonekab"
                )
        self.device = device
        self.languages = ["en", "fr"]
        self.preprocess = get_dino_preprocess()

        # Load cached embeddings
        cache_path = os.path.join(embedding_cache_dir, f"cached_flickr_feats_{split}.pt")
        self.embedding_cache = torch.load(cache_path)
        self.return_raw = return_raw

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        cached = self.embedding_cache[idx]

        # Randomly choose language
        caption_en = sample["en"]
        caption_fr = sample["fr"]
        lang_idx = random.choice([0, 1])  # 0 = English, 1 = French
        caption = caption_en if lang_idx == 0 else caption_fr
        text_feat = F.normalize(cached["text_feat_en"] if lang_idx == 0 else cached["text_feat_fr"], dim=0)
        other_text_feat = F.normalize(cached["text_feat_fr"] if lang_idx == 0 else cached["text_feat_en"], dim=0)

        image_feat = F.normalize(cached["image_feat"], dim=0)

        if self.return_raw:
            # Get raw image and caption text
            image = self.preprocess(sample["image"])  # Not used in training, just for x1
            return image_feat, [cached["text_feat_en"], cached["text_feat_fr"]], image, [caption_en, caption_fr], lang_idx
        else:
            return image_feat, [cached["text_feat_en"], cached["text_feat_fr"]], lang_idx


def main(lr, bs, rank, prune_th, seed_id,
         embedding_cache_dir="/data/stonekab/",
         cache_name="urfunny_feats.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Reproducibility
    # random.seed(42 + seed_id)
    # torch.manual_seed(42 + seed_id)
    # if device == "cuda":
    #     torch.cuda.manual_seed_all(42 + seed_id)

    train_dataset = Multi30KMixedLangDataset(split="train")
    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    val_dataset = Multi30KMixedLangDataset(split="validation")

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")

    # en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # fr_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)
    
    # Infer dims from dataset (don’t hardcode 768/1024)
    in_dims = [768,768]

    projection_model = MultiLoReFT(
        input_dims=in_dims,
        shared_rank=rank,
        specific_rank=rank,
        pruning_threshold=prune_th,
        device=device,
        staging=True,
        pruning=True,
        shared_R_mode="pad",
        dataset_name="flickr",
    ).to(device)

    early_stopping_config = {
        "shared":  {"patience": 20,  "min_improvement_ratio": 0.001, "max_epochs": 300},
        "private": {"patience": 20,  "min_improvement_ratio": 0.001, "max_epochs": 300},
        "joint":   {"patience": 150, "min_improvement_ratio": 0.001, "max_epochs": 6000},
    }

    projection_model.train_projection(
        train_dataloader,
        val_dataloader,
        early_stopping_config,
        lr=lr,
        epochs=6000,
        exp_name=f"multi_loreft_lr{lr:.4f}_bs{bs}_rank{rank}_prune{prune_th:.2f}_{seed_id}",
        dataset_name="flickr",
        en_tokenizer=None,
        fr_tokenizer=None,
    )


if __name__ == "__main__":
    # test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # image_encoder = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
    # english_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
    # french_encoder = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)

    ranks = [700]
    for lr in [1e-3]:
        for bs in [256]:
            for rank in ranks:
                for prune_th in [0.1]:
                    run_name = 'flickr_multi_loreft_lr%.4f_bs%d_rank%d_prune%.2f'%(lr, bs, rank, prune_th)
                    wandb.init(project="MultiLoReFT", name=run_name, config={"lr": lr, "batch": bs, "rank": rank, "prune_th": prune_th})
                    for seed_id in range(3):
                        main(lr, bs, rank, prune_th, seed_id)
                    wandb.finish()