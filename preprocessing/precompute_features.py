import torch
from datasets import load_dataset
# from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from transformers import AutoTokenizer, AutoModel  # For both LLaMA and LaBSE
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import argparse

# ====== Args ======
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['flickr', 'vqa'], required=True)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = f"/data/stonekab/cached_{args.dataset}_feats"
os.makedirs(save_dir, exist_ok=True)

# ====== Load Encoders ======
image_encoder = timm.create_model('vit_base_patch14_dinov2', pretrained=True).to(device)
image_encoder.eval()

# ===== [REPLACED] English BERT Encoder =====
# english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# english_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
# english_encoder.eval()

# ===== English Lightweight Llama Encoder =====
# Using LLaMA (tiny) for English text encoding
english_tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')  # Example lightweight Llama tokenizer, replace with actual model as needed
english_encoder = AutoModel.from_pretrained('NousResearch/Llama-2-7b-hf').to(device)      # Use a SMALLER Llama if available
english_encoder.eval()

# French text encoder remains the same for now (LaBSE)
french_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
french_encoder = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)
french_encoder.eval()

# ====== Preprocessing ======
def get_dino_preprocess(image_size=518):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

preprocess = get_dino_preprocess()

# ====== Main Processing Function ======
def process_dataset(split):
    print(f"\nProcessing split: {split}")
    if args.dataset == "flickr":
        dataset_name = "romrawinjp/multi30k"
        hf_split = split
    else:
        dataset_name = "HuggingFaceM4/VQAv2"
        hf_split = split  # split names like 'train', 'validation', 'test'
        # hf_split = split if args.dataset == "flickr" else f"vqa_{split}"
    dataset = load_dataset(dataset_name, split=hf_split, cache_dir="/data/stonekab")

    all_data = []
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset.select(range(i, min(i + args.batch_size, len(dataset))))
        images = []
        raw_texts_en = []
        raw_texts_fr = [] if args.dataset == "flickr" else None
        answers = []

        for sample in batch:
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(preprocess(img))

            if args.dataset == "flickr":
                raw_texts_en.append(sample["en"])
                raw_texts_fr.append(sample["fr"])
            else:
                raw_texts_en.append(sample["question"])
                answers.append(sample["multiple_choice_answer"])

        # === Encode images ===
        image_tensor = torch.stack(images).to(device)  # [B, 3, 518, 518]
        with torch.no_grad():
            image_feat = image_encoder.forward_features(image_tensor)[:, 0, :].cpu()

        # === Encode English Text ===
        # tokens_en = english_tokenizer(raw_texts_en, return_tensors="pt", truncation=True, padding=True).to(device)
        # with torch.no_grad():
        #     text_feat_en = english_encoder(**tokens_en).last_hidden_state[:, 0, :].cpu()

        tokens_en = english_tokenizer(raw_texts_en, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            text_outputs_en = english_encoder(**tokens_en)
            if hasattr(text_outputs_en, "last_hidden_state"):
                text_feat_en = text_outputs_en.last_hidden_state[:, 0, :].cpu()
            else:  # Llama or similar sometimes just returns one tensor
                text_feat_en = text_outputs_en[0][:, 0, :].cpu()

        # === Encode French Text === (Flickr only)
        if args.dataset == "flickr":
            tokens_fr = french_tokenizer(raw_texts_fr, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                text_feat_fr = french_encoder(**tokens_fr).last_hidden_state[:, 0, :].cpu()
        else:
            # Encode VQA answer (as target) using same BERT encoder
            tokens_ans = english_tokenizer(answers, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                answer_feat = english_encoder(**tokens_ans).last_hidden_state[:, 0, :].cpu()

        # === Store ===
        for j in range(len(batch)):
            sample_data = {
                "index": i + j,
                "split": split,
                "image_feat": image_feat[j],
                "text_feat_en": text_feat_en[j],
            }
            if args.dataset == "flickr":
                sample_data["caption_en"] = raw_texts_en[j]
                sample_data["caption_fr"] = raw_texts_fr[j]
                sample_data["text_feat_fr"] = text_feat_fr[j]
            else:
                sample_data["question"] = raw_texts_en[j]
                sample_data["answer"] = answers[j]
                sample_data["answer_feat"] = answer_feat[j]

            all_data.append(sample_data)

    save_path = os.path.join(save_dir, f"cached_{args.dataset}_feats_{split}.pt")
    torch.save(all_data, save_path)
    print(f"Saved {len(all_data)} items to {save_path}")

# ====== Run for Each Split ======
splits = ["train", "validation"]#, "test"]
for split in splits:
    process_dataset(split)
