import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
# import clip
import os
import random
# from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from multimodal_projector import MultiLoReFT
import torch.nn.functional as F
import timm
import sys


def get_dino_preprocess(image_size=518):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

class Multi30KMixedLangDataset(Dataset):
    def __init__(self, split='train', device='cuda', embedding_cache_dir="/data/stonekab/cached_flickr_feats"):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        cached = self.embedding_cache[idx]

        # Get raw image and caption text
        image = self.preprocess(sample["image"])  # Not used in training, just for x1
        caption_en = sample["en"]
        caption_fr = sample["fr"]

        # Randomly choose language
        lang_idx = random.choice([0, 1])  # 0 = English, 1 = French
        caption = caption_en if lang_idx == 0 else caption_fr
        text_feat = cached["text_feat_en"] if lang_idx == 0 else cached["text_feat_fr"]

        image_feat = cached["image_feat"]

        # Return: h1 (image emb), h2 (text emb), x1 (raw image), x2 (caption str), label (0 or 1)
        return image_feat.to(self.device), [cached["text_feat_en"].to(self.device), cached["text_feat_fr"].to(self.device)], image.to(self.device), [caption_en, caption_fr], lang_idx


if __name__ == "__main__":
    # log_file = open("flickr_train.log", "w")
    # sys.stdout = log_file
    # sys.stderr = log_file  # Optional: also log errors

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Running on CPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create train/val/test splits
    train_dataset = Multi30KMixedLangDataset(split="train", device=device)
    train_dataset = torch.utils.data.Subset(train_dataset, range(4000))
    val_dataset = Multi30KMixedLangDataset(split="validation", device=device)
    # test_dataset = Multi30KMixedLangDataset(split="test", device=device)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    # print(f"Test size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # image_encoder = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
    # english_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
    # french_encoder = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)

    projection_model = MultiLoReFT(
        input_dims=[768,768], 
        shared_rank=128, 
        specific_rank=128, 
        pruning_threshold=1.,
        device=device,
        staging=True,
        pruning=True,
        dataset_name="flickr"
    ).to(device)
    

    # for image_feats, text_feats, images, captions, labels in train_dataloader:
    #     print("Image features:", image_feats.shape)  # (B, D)
    #     print("Text features:", text_feats.shape)    # (B, D)  )
    #     print(labels)
    #     print("Caption:", captions[0])  
    #     # Display first image in batch
    #     plt.figure(figsize=(10,10))
    #     img = images[0].permute(1,2,0).cpu().numpy()
    #     # Denormalize image
    #     img = (img * 0.5) + 0.5
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.savefig('./plots/test.png')
    #     plt.show()
    #     break
    # Train model
        # Early stopping configuration
    early_stopping_config = {
        "shared": {
            "patience": 5,
            "min_improvement_ratio": 0.001,
            "max_epochs": 40
        },
        "private": {
            "patience": 5,
            "min_improvement_ratio": 0.001,
            "max_epochs": 40
        },
        "joint": {
            "patience": 5,
            "min_improvement_ratio": 0.001,
            "max_epochs": 2000
        }
    }

    projection_model.train_projection(train_dataloader, val_dataloader, early_stopping_config, lr=1e-3, epochs=60, exp_name='flickr_model_all')#, save_path='./ckpts/flickr_model_staging.pth')