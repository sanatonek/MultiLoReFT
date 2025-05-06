import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import clip
import random
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torch.utils.data import DataLoader


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

        caption = random.choice(sample[f"sentence_{lang}"])  # one of the 5 captions

        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image.unsqueeze(0)).squeeze(0).cpu()

            if lang == "en":
                tokens = clip.tokenize([caption]).to(self.device)
                text_feat = self.clip_model.encode_text(tokens).squeeze(0).cpu()
            else:  # "fr"
                text_feat = self.encode_text_fr([caption]).squeeze(0).cpu()

        return image_feat, text_feat, lang_idx, image, caption  # label: 0 for EN, 1 for FR


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Multi30KMixedLangDataset(split="train", device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for image_feats, text_feats, lang_labels, images, captions in dataloader:
        print("Image features:", image_feats.shape)  # (B, D)
        print("Text features:", text_feats.shape)    # (B, D)
        print("Language labels:", lang_labels)       # (B,)
        print("Caption:", captions[0])
        break