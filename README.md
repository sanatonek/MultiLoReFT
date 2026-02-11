# MultiLoReFT

**Low-Rank Finetuning for Multimodal Representation Learning**

MultiLoReFT learns to decompose multimodal embeddings into **shared** and **modality-specific** subspaces using low-rank projection matrices. The model enables downstream analysis of what information is common across modalities (e.g., emotion in video+audio) versus what is unique to each (e.g., visual style, acoustic texture).

---

## The Model

MultiLoReFT projects two modality embeddings \( h_1, h_2 \) through learnable low-rank matrices:

- **Shared subspace** (\( R_s \)): Captures information that is aligned across modalities (e.g., sentiment, identity, content).
- **Modality-specific subspaces** (\( R_{m1}, R_{m2} \)): Capture information unique to each modality.

The model is trained with:
- **Orthogonality loss**: Keeps shared and modality-specific subspaces distinct.
- **Independence loss**: Encourages shared and modality-specific representations to be statisticallly independant.
- **Mutual information loss**: Balances shared vs. modality-specific information.

**Staged training** is used: first train the shared subspace, then the modality-specific subspaces, then jointly. Optional **singular-value pruning** removes weak dimensions during training to adapt rank automatically.

**Output representations**:
- `decouple()`: Returns \( (z_m, z_s) \) per modality — modality-specific and shared components.
- `fuse_representations()`: Concatenates \( z_{m1}, z_{m2}, \bar{z}_s \) for downstream tasks.

---

## Project Structure

```
multimodal_LoReFT/
├── configs/
│   ├── datasets.yaml      # Dataset-specific configuration (model params, paths, tasks)
│   └── load.py            # Config loading utilities
├── src/
│   ├── multimodal_projector.py   # MultiLoReFT model
│   ├── losses.py                 # Orthogonality, independence, MI losses
│   ├── utils.py                  # Checkpoint loading, SklearnTrainer
│   ├── data/
│   │   └── base.py               # MultimodalDataset (h1, h2, x1, x2, labels)
│   └── baselines/                # Apollo, contrastive, DRIM baselines
├── scripts/
│   ├── simulation.py             # Train MultiLoReFT on simulated/simulated_apollo
│   ├── cremad.py                 # CremadDataset + train on CREMA-D
│   ├── flickr.py                 # Multi30KMixedLangDataset (Flickr30k)
│   ├── urfunny.py                # UrFunnyDataset (humor detection)
│   ├── vqa.py                    # VQADataset (VQA v2)
│   ├── evaluate_representations.py   # Evaluate predictability of shared vs modality-specific
│   └── baselines/                # Launchers for Apollo, contrastive, DRIM
├── preprocessing/            # Feature extraction scripts
└── data/                     # Simulated .npz files (gitignored)
```

**Design principle**: Each dataset has a dedicated script (e.g., `cremad.py`, `flickr.py`) that defines a `Dataset` class yielding `(h1, h2, ...)` where `h1` and `h2` are precomputed feature tensors for the two modalities. All datasets share the same `MultiLoReFT` model and training loop structure.

---

## Quick Start

Run all commands from the **project root**:

```bash
cd /path/to/multimodal_LoReFT
python scripts/simulation.py --dataset simulated
python scripts/evaluate_representations.py --dataset simulated
```

---

## Workflow

### 1. Preprocess (if needed)

For raw datasets, extract features first. Use any pre-trained foundation model appropriate for the modality to extract the unimodal representations.

```bash
# Example: CREMA-D (video + audio)
python preprocessing/cremad_process_2.py

# UrFunny, VQA, etc.
python preprocessing/urfunny_process.py
python preprocessing/vqa_process.py
```

### 2. Train

Run the dataset-specific script. Configuration is centralized in `configs/datasets.yaml`, but training scripts may read hyperparameters from there or use defaults.

```bash
# Simulated data
python scripts/simulation.py --dataset simulated

# CREMA-D (emotion from video + audio)
python scripts/cremad.py

# Flickr30k (multilingual image–caption)
python scripts/flickr.py

# UrFunny (humor from video + text)
python scripts/urfunny.py
```

Checkpoints are saved to `./ckpts/` with names derived from the config (see `checkpoint_pattern` in `configs/datasets.yaml`).

### 3. Evaluate

`scripts/evaluate_representations.py` loads a trained model, extracts shared and modality-specific representations, and evaluates how well each predicts downstream labels (e.g., emotion, humor, language).

```bash
python scripts/evaluate_representations.py --dataset simulated
python scripts/evaluate_representations.py --dataset cremad
python scripts/evaluate_representations.py --dataset flickr

# Optional: benchmark against baselines (attention fusion, MIA)
python scripts/evaluate_representations.py --dataset cremad --baselines

# Optional: compare to CLIP features (Flickr only)
python scripts/evaluate_representations.py --dataset flickr --contrastive
```

The script:
1. Loads the checkpoint for the specified dataset.
2. Runs data through the model and calls `decouple()` to get \( z_s, z_{m1}, z_{m2} \).
3. For each task in `task_names`, trains a simple classifier/regressor on each component (shared, modality-specific, concatenated) and reports predictability (e.g., ROC-AUC, MSE).
4. Saves 2D PCA plots to `./plots/<dataset>/`.

---

## Adding a New Dataset

To run MultiLoReFT on a new dataset:

### 1. Add configuration in `configs/datasets.yaml`

```yaml
my_dataset:
  data_path: "./data/my_dataset.npz"   # or omit if loading from Dataset class
  task_names:
    - task_a
    - task_b
  modality_names:
    - "modality_1"
    - "modality_2"
  input_dims: [dim1, dim2]
  shared_rank: 128
  specific_rank: 128
  rank: 128
  pruning_threshold: 0.1
  batch_size: 256
  lr: 0.001
  prune: 0.1
  checkpoint_pattern: "{ckpts_dir}/{dataset}_multi_loreft_lr{lr:.4f}_bs{bs}_rank{rank}_prune{prune:.2f}_{seed}_no_stage.pth"
```

### 2. Create a dataset script in `scripts/`

Create `scripts/my_dataset.py` with a `torch.utils.data.Dataset` that returns, for each sample:

- `h1`: Feature tensor for modality 1 (e.g., video) — shape `(batch, dim1)`
- `h2`: Feature tensor for modality 2 (e.g., audio) — shape `(batch, dim2)`
- Labels for each task in `task_names`

Follow the pattern of `scripts/cremad.py` or `scripts/flickr.py`:

```python
class MyDataset(Dataset):
    def __init__(self, split='train', ...):
        # Load precomputed features or compute on the fly
        self.mod1_dim = ...
        self.mod2_dim = ...

    def __getitem__(self, idx):
        return h1, h2, x1, x2, label1, label2, ...
```

### 3. Add a training script

Create or extend a script (e.g., `scripts/my_dataset_train.py`) that:

1. Instantiates `MyDataset` for train/val/test.
2. Builds `MultiLoReFT` with `input_dims=[mod1_dim, mod2_dim]` and other params from config.
3. Trains with the staged protocol (shared → private → joint) and optional pruning.
4. Saves checkpoints using the same pattern as other datasets.

### 4. Extend `evaluate_representations.py`

In `main()`, add a branch for your dataset:

```python
elif dataset_name == "my_dataset":
    from scripts.my_dataset import MyDataset
    train_dataset = MyDataset(split="train")
    test_dataset = MyDataset(split="test")
    input_dims = [train_dataset.mod1_dim, train_dataset.mod2_dim]
    projection_model = MultiLoReFT(
        input_dims=input_dims,
        shared_rank=cfg["shared_rank"],
        specific_rank=cfg["specific_rank"],
        ...
    ).to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], ...)
    test_dataloader = DataLoader(test_dataset, ...)
```

Then add a `load_components()` branch to extract `(h1, h2)` and labels from your dataloader batches so the evaluation logic can run.



## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `numpy`, `scikit-learn`, `PyYAML`, `matplotlib`, `seaborn`. See `requirements.txt` for versions.

