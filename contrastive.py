import torch
from torch.utils.data import DataLoader
from flickr import Multi30KMixedLangDataset
import clip
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from evaluate_representations import evaluate_cross_modal_retrieval, plot_representations, SimilarityMLP
from torchvision.transforms.functional import to_pil_image

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load CLIP (for English)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Load M-CLIP (for French)
# mclip_model = SentenceTransformer("M-CLIP/XLM-Roberta-Large-Vit-B-16Plus").to(device)
# mclip_model.eval()

# Dataset
dataset = Multi30KMixedLangDataset(split="test", device=device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

all_image_embs = []
all_text_embs = []
langs = []

# For plotting
raw_images_all = []
captions_all = []

with torch.no_grad():
    for batch in dataloader:
        image_feats, text_feats_all, raw_images, captions_batch, lang_labels = batch

        for i in range(len(raw_images)):
            lang = 0
            # lang = torch.randint(0, 2, (len(image_feats),))
            # Use the labels to select language for each item
            caption = captions_batch[0][i] if lang_labels[i] == 0 else captions_batch[1][i]
            # lang = lang_labels[i].item()
            # caption = captions_batch[0][i] if lang == 0 else captions_batch[1][i]
            langs.append("en" if lang == 0 else "fr")
            raw_images_all.append(raw_images[i])
            captions_all.append(caption)

            # Image encoding
            img_pil = to_pil_image(raw_images[i].cpu())
            img_input = clip_preprocess(img_pil).unsqueeze(0).to(device)
            # img_input = clip_preprocess(raw_images[i]).unsqueeze(0).to(device)
            image_emb = clip_model.encode_image(img_input)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            # Text encoding
            if lang == 0:
                text_tokens = clip.tokenize([caption]).to(device)
                text_emb = clip_model.encode_text(text_tokens)
            else:
                text_emb = mclip_model.encode([caption], convert_to_tensor=True).to(device)

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            all_image_embs.append(image_emb.squeeze(0).cpu())
            all_text_embs.append(text_emb.squeeze(0).cpu())

# Stack all
all_image_embs = torch.stack(all_image_embs).float()
all_text_embs = torch.stack(all_text_embs).float()

print("Image embeddings shape:", all_image_embs.shape)
print("Text embeddings shape:", all_text_embs.shape)
print("Languages example:", langs[:10])

# === Evaluate cross-modal retrieval ===
res = evaluate_cross_modal_retrieval(all_text_embs, all_image_embs, device, batch_size=256, similarity_model=SimilarityMLP(all_text_embs.shape[1], all_image_embs.shape[1]))
print("Recall@10 predicting image from text: ", res)
res = evaluate_cross_modal_retrieval(all_image_embs, all_text_embs, device, batch_size=256, similarity_model=SimilarityMLP(all_image_embs.shape[1], all_text_embs.shape[1]))
print("Recall@10 predicting caption from image: ", res)


# === Plot representations (optional) ===
labels_np = np.array([0 if l == "en" else 1 for l in langs])
plot_representations(
    (all_image_embs, all_image_embs, all_text_embs, all_text_embs),
    labels_np,
    task_name="language_contrastive",
    dataset_name="flickr",
    save_dir="./plots"
)

print("Done!")
