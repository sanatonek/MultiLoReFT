import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import clip
import random
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from multimodal_projector import ProjectionModule, train
import torch.nn.functional as F

class Multi30KMixedLangDataset(Dataset):
    def __init__(self, split='train', device='cuda'):
        self.dataset = load_dataset("romrawinjp/multi30k", split=split)
        self.device = device
        self.languages = ["en", "fr"]  # only using EN and FR

        # Load CLIP (English only)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        # Load multilingual encoder for French
        self.lang_model_name = "sentence-transformers/LaBSE"
        self.tokenizer = AutoTokenizer.from_pretrained(self.lang_model_name)
        self.lang_model = AutoModel.from_pretrained(self.lang_model_name).to(self.device)
        self.lang_model.eval()

        # Normalize transform (same as CLIP default)
        self.image_transform = self.clip_preprocess

    def encode_text_fr(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.lang_model(**tokens)
            embeddings = model_output.last_hidden_state[:, 0, :]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.image_transform(sample["image"]).to(self.device)  # (3, H, W)

        # Randomly choose English (0) or French (1)
        lang_idx = random.choice([0, 1])
        lang = self.languages[lang_idx]

        caption_en = sample["en"]  # one of the 5 captions
        caption_fr = sample["fr"]  # one of the 5 captions
        caption = [caption_en, caption_fr][lang_idx]
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image.unsqueeze(0)).squeeze(0).cpu()

            tokens = clip.tokenize([caption_en]).to(self.device)
            if lang == "en":
                caption = caption_en
                text_feat = self.clip_model.encode_text(tokens).squeeze(0).cpu()
            else: ## Work on a better strategy later!
                caption = caption_fr
                text_feat = self.encode_text_fr([caption_fr]).squeeze(0).cpu()
                # Ensure same dimensionality as CLIP text features
                if text_feat.shape != self.clip_model.encode_text(tokens).squeeze(0).shape:
                    text_feat = F.interpolate(text_feat.unsqueeze(0).unsqueeze(0), 
                                            size=self.clip_model.encode_text(tokens).squeeze(0).shape[0],
                                            mode='linear').squeeze(0).squeeze(0)
# return image_feat, [text_feat_en, text_feat_fr], image, [caption_en, caption_fr]
        return image_feat, text_feat, image, caption, [lang_idx]  # label: 0 for EN, 1 for FR


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Running on CPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create train/val/test splits
    train_dataset = Multi30KMixedLangDataset(split="train", device=device)
    val_dataset = Multi30KMixedLangDataset(split="validation", device=device)
    test_dataset = Multi30KMixedLangDataset(split="test", device=device)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    projection_model = ProjectionModule(
        input_dims=[512,512], 
        shared_rank=256, 
        specific_rank=256, 
        data_dim=None
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(projection_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    

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
    train(projection_model, train_dataloader, val_dataloader, optimizer, device, scheduler, epochs=200)