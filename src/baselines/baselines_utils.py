"""Shared utilities for baseline models: data loading, evaluation, and CLI."""
import sys
import os as _os
import torch
import numpy as np
from torch.utils.data import DataLoader

_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_os.path.dirname(_script_dir))


def ensure_project_path():
    """Ensure project root is on sys.path for imports."""
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)


CREMAD_TASK_NAMES = ["subject_id", "sentence_id", "emotion", "age", "sex", "race", "ethnicity"]


def gather_samples(dataloader, dataset_name, task_filter=None):
    """Collect modality features and labels from a dataloader.

    Args:
        dataloader: DataLoader to iterate
        dataset_name: 'cremad', 'urfunny', or 'flickr'
        task_filter: optional list of task names to keep (e.g. ['emotion'] for cremad)

    Returns:
        prediction_labels: list of tensors, one per task
        task_names: list of task names
        h1, h2: stacked features from modality A and B
    """
    if dataset_name == "cremad":
        n_tasks = 7
        task_names = CREMAD_TASK_NAMES
    elif dataset_name == "urfunny":
        n_tasks = 1
        task_names = ["humor"]
    elif dataset_name == "flickr":
        raise ValueError("Use get_flickr_batch for flickr")
    else:
        raise ValueError(f"gather_samples not implemented for {dataset_name}")

    prediction_labels = [[] for _ in range(n_tasks)]
    h1_list, h2_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            if dataset_name == "cremad":
                video_feats, audio_feats, x1, x2, subject_id, sentence_id, emotion, age, sex, race, ethnicity = batch
                sentence_refs = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]
                emotion_refs = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
                subject_id = torch.tensor([int(id) for id in subject_id], dtype=torch.float32)
                sentence_id = torch.tensor([sentence_refs.index(id) for id in sentence_id], dtype=torch.float32)
                emotion = torch.tensor([emotion_refs.index(id) for id in emotion], dtype=torch.float32)
                h1_list.append(video_feats)
                h2_list.append(audio_feats)
                prediction_labels[0].append(subject_id)
                prediction_labels[1].append(sentence_id)
                prediction_labels[2].append(emotion)
                prediction_labels[3].append(age)
                prediction_labels[4].append(sex)
                prediction_labels[5].append(race)
                prediction_labels[6].append(ethnicity)
            elif dataset_name == "urfunny":
                video_feats, audio_feats, x1, x2, humor = batch
                h1_list.append(video_feats)
                h2_list.append(audio_feats)
                prediction_labels[0].append(humor)
    h1 = torch.cat(h1_list, dim=0)
    h2 = torch.cat(h2_list, dim=0)
    for i in range(len(prediction_labels)):
        prediction_labels[i] = torch.cat(prediction_labels[i], dim=0)

    if task_filter is not None:
        indices = [task_names.index(t) for t in task_filter if t in task_names]
        prediction_labels = [prediction_labels[i] for i in indices]
        task_names = task_filter

    return prediction_labels, task_names, h1, h2


def run_evaluation(components, components_test, prediction_labels, prediction_labels_test,
                  task_names, dataset_name):
    """Run evaluate_predictability for each task and return aggregated results dict."""
    from scripts.evaluate_representations import evaluate_predictability

    results_dict = {n: {} for n in task_names}
    for task_ind, label_task in enumerate(prediction_labels):
        label_task = label_task.squeeze()
        results_dict[task_names[task_ind]] = evaluate_predictability(
            components, label_task, task_names[task_ind],
            dataset_name=dataset_name,
            components_test=components_test,
            labels_test=prediction_labels_test[task_ind],
        )
    return results_dict


def aggregate_seed_results(results_across_seeds, results_dict):
    """Merge results_dict from one seed into results_across_seeds."""
    if not results_across_seeds:
        results_across_seeds = {n: {k: [] for k in results_dict[n].keys()} for n in results_dict}
    for label_name, result in results_dict.items():
        for task_name in result.keys():
            results_across_seeds[label_name][task_name].extend(result[task_name])
    return results_across_seeds


def print_aggregated_results(results_across_seeds):
    """Print mean ± std for each task and component."""
    for label_name, results in results_across_seeds.items():
        print("Predicting label:", label_name)
        for task_name in results.keys():
            mean_score = np.mean(results[task_name], axis=0)
            var_score = np.var(results[task_name], axis=0)
            print(f"  Component: {task_name}: {mean_score:.3f} ± {np.sqrt(var_score):.3f}")
        print("--------------------------------")


def get_baseline_parser(description):
    """Return parser with standard baseline args: --dataset_name, --train, --ckpt_dir."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["simulated", "simulated_apollo", "cremad", "urfunny", "flickr"],
        default="simulated",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory")
    return parser


def parse_baseline_args(description):
    """Parse standard baseline args."""
    return get_baseline_parser(description).parse_args()


def get_flickr_batch(dataloader, lang_idx_key=-1):
    """Build batch dict and prediction_labels from Flickr dataloader."""
    train_batch = {"A": [], "B": []}
    prediction_labels = [[]]
    for batch in dataloader:
        image_feats, caption_feats, x1, x2, lang_idx = batch
        train_batch["A"].append(image_feats)
        train_batch["B"].append(torch.stack([caption_feats[lang_idx[i]][i] for i in range(len(image_feats))], dim=0))
        prediction_labels[0].append(lang_idx)
    train_batch["A"] = torch.cat(train_batch["A"], dim=0)
    train_batch["B"] = torch.cat(train_batch["B"], dim=0)
    prediction_labels[0] = torch.cat(prediction_labels[0], dim=0)
    return train_batch, prediction_labels
