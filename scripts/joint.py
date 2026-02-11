import sys
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np

from scripts.cremad import CremadDataset
from src.multimodal_projector import MultiLoReFT
from src.utils import load_checkpoint

# Define a simple 2-layer MLP
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets with different fusion methods.")
    parser.add_argument('--fusion', type=str, default="concat")
    parser.add_argument('--dataset', type=str, default="cremad")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if args.dataset == "cremad":
        train_dataset = CremadDataset(split='train')
        val_dataset = CremadDataset(split='val')
        label_index = 1
        input_dim = 1280
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)

    from sklearn.linear_model import LogisticRegression
    from src.utils import SklearnTrainer

    # Prepare the data
    # train_features = []
    # train_labels = []
    # for batch in train_dataloader:
    #     video_feats, audio_feats, _, _, subject_id, sentence_id, emotion, age, sex, race, ethnicity = batch
    #     fused_representation = torch.cat((video_feats, audio_feats), dim=1).cpu().numpy()
    #     train_features.append(fused_representation)
    #     emotion_to_num = {emotion: idx for idx, emotion in enumerate(sorted(set(emotion)))}
    #     train_labels.append([emotion_to_num[e] for e in emotion])

    # train_features = np.concatenate(train_features, axis=0)
    # train_labels = np.concatenate(train_labels, axis=0)
    if args.fusion == "multiloreft":
        projection_model = MultiLoReFT(
            input_dims=[400,768], 
            shared_rank=768, 
            specific_rank=768,
            device=device,
            shared_R_mode="pad",
            dataset_name="cremad")
        projection_model = load_checkpoint(filepath="./ckpts/cremad_model_all.pth", model=projection_model)
        projection_model.eval()
        projection_model.to(device)

    val_features = []
    val_labels = []
    for batch in val_dataloader:
        video_feats, audio_feats, _, _, subject_id, sentence_id, emotion, age, sex, race, ethnicity = batch
        if args.fusion == "concat":
            fused_representation = torch.cat((video_feats, audio_feats), dim=1).cpu().numpy()
        elif args.fusion == "1":
            fused_representation = video_feats.cpu().numpy()
        elif args.fusion == "2":
            fused_representation = audio_feats.cpu().numpy()    
        elif args.fusion == "multiloreft":
            phis = projection_model([video_feats.to(device), audio_feats.to(device)])
            fused_representation = projection_model.fuse_representations(phis).detach().cpu().numpy()
        val_features.append(fused_representation)
        emotion_to_num = {emotion: idx for idx, emotion in enumerate(sorted(set(emotion)))}
        val_labels.append([emotion_to_num[e] for e in emotion])

    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Initialize and train the model using SklearnTrainer
    model = LogisticRegression(max_iter=1000)
    
    trainer = SklearnTrainer(model, task_type="multiclass")
    acc = trainer.train_and_evaluate(val_features , val_labels, k=3)
    print(f"Accuracy: {acc[0]} +- {acc[1]}")