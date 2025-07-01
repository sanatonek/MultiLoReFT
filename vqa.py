import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
import random
import torch.nn.functional as F
from transformers import BertTokenizer
from multimodal_projector import MultiLoReFT
from utils import get_dino_preprocess

class VQADataset(Dataset):
    def __init__(self, split='train', device='cuda', embedding_cache_dir="/data/stonekab/cached_vqa_feats"):
        self.dataset = load_dataset("HuggingFaceM4/VQAv2", split=split, cache_dir="/data/stonekab")
        self.device = device
        cache_path = os.path.join(embedding_cache_dir, f"cached_vqa_feats_{split}.pt")
        self.embedding_cache = torch.load(cache_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        cached = self.embedding_cache[idx]

        question = sample["question"]
        answer = sample["multiple_choice_answer"]
        img = sample["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        img_tensor = get_dino_preprocess()(img)

        question_feat = F.normalize(cached["text_feat_en"], dim=0)
        answer_feat = F.normalize(cached["answer_feat"], dim=0)
        image_feat = F.normalize(cached["image_feat"], dim=0)
        answer_feat = F.normalize(cached["answer_feat"], dim=0)
        return image_feat, question_feat, img_tensor, question, answer, answer_feat

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_dataset = VQADataset(split="train")
    train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
    val_dataset = VQADataset(split="validation")
    val_dataset = torch.utils.data.Subset(val_dataset, range(1000))

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    projection_model = MultiLoReFT(
        input_dims=[768, 768],
        shared_rank=768,
        specific_rank=768,
        pruning_threshold=0.2,
        device=device,
        staging=True,
        pruning=True,
        dataset_name="vqa"
    ).to(device)

    early_stopping_config = {
        "shared": {"patience": 50, "min_improvement_ratio": 0.001, "max_epochs": 100},
        "private": {"patience": 50, "min_improvement_ratio": 0.001, "max_epochs": 100},
        "joint": {"patience": 50, "min_improvement_ratio": 0.001, "max_epochs": 2000}
    }

    projection_model.train_projection(
        train_dataloader,
        val_dataloader,
        early_stopping_config,
        lr=1e-3,
        epochs=250,
        exp_name='vqa_model_all',
        en_tokenizer=en_tokenizer,
        fr_tokenizer=None  # Not used for VQA
    )
