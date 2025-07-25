import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

class SimilarityMLP(torch.nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=256):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2),
        )
    def forward(self, x1):
        score = self.fc(x1)
        return score

def evaluate_cross_modal_retrieval(h0, h1, device, batch_size=512, similarity_model=None, k=10):
    """
    Batched version to evaluate cross-modal retrieval with learned similarity.
    similarity_model: a model taking (query, gallery) → score
    """
    h0 = h0.to(device)
    h1 = h1.to(device)
    similarity_model = similarity_model.to(device)

    def recall_at_k_batched(query_set, gallery_set, k=10):
        correct_count = 0
        num_samples = query_set.shape[0]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_query = query_set[start:end]  # [B, Dq]
            if batch_query.size(1) < gallery_set.size(1):
                padding = torch.zeros(batch_query.size(0), gallery_set.size(1) - batch_query.size(1), device=batch_query.device)
                batch_query = torch.cat((batch_query, padding), dim=1)
            elif batch_query.size(1) > gallery_set.size(1):
                padding = torch.zeros(gallery_set.size(0), batch_query.size(1) - gallery_set.size(1), device=gallery_set.device)
                gallery_set = torch.cat((gallery_set, padding), dim=1)
            #projected_query = similarity_model(batch_query)
            sim_matrix = torch.nn.functional.cosine_similarity(batch_query.unsqueeze(1), gallery_set, dim=2)
            topk = sim_matrix.topk(k, dim=1).indices
            true_matches = torch.arange(start, end, device=device).unsqueeze(1)
            correct = (topk == true_matches).any(dim=1).float()
            correct_count += correct.sum().item()

        return correct_count / num_samples
    return recall_at_k_batched(h0, h1, k)

def calc_corrs_and_ranks(model, threshold=0.05):
    # Get matrices above threshold
    Rs = model.R_s.detach().cpu().numpy()
    Rm1 = model.R_m1.detach().cpu().numpy()
    Rm2 = model.R_m2.detach().cpu().numpy()
    matrices = [Rs, Rm1, Rm2]
    corr_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            flat_i = matrices[i].flatten()
            flat_j = matrices[j].flatten()
            if len(flat_i) > len(flat_j):
                flat_j = np.pad(flat_j, (0, len(flat_i) - len(flat_j)))
            elif len(flat_j) > len(flat_i):
                flat_i = np.pad(flat_i, (0, len(flat_j) - len(flat_i)))
            corr_matrix[i,j] = np.corrcoef(flat_i, flat_j)[0,1]
    
    #shared_sv = len(torch.where(torch.linalg.svdvals(model.R_s) > threshold)[0].tolist())
    # making sure we get the max rank possible no matter whether init rank or input dim are larger
    shared_sv = max(len(torch.where(torch.linalg.svdvals(model.R_s) > threshold)[0].tolist()), len(torch.where(torch.linalg.svdvals(model.R_s.T) > threshold)[0].tolist()))
    m1_sv = max(len(torch.where(torch.linalg.svdvals(model.R_m1) > threshold)[0].tolist()), len(torch.where(torch.linalg.svdvals(model.R_m1.T) > threshold)[0].tolist()))
    m2_sv = max(len(torch.where(torch.linalg.svdvals(model.R_m2) > threshold)[0].tolist()), len(torch.where(torch.linalg.svdvals(model.R_m2.T) > threshold)[0].tolist()))

    out_dict = {
        'corr R_s-R_m1': corr_matrix[0,1],
        'corr R_s-R_m2': corr_matrix[0,2],
        'corr R_m1-R_m2': corr_matrix[1,2],
        'rank R_s': shared_sv,
        'rank R_m1': m1_sv,
        'rank R_m2': m2_sv,
        'mean R_s': np.mean(Rs),
        'mean R_m1': np.mean(Rm1),
        'mean R_m2': np.mean(Rm2),
        'std R_s': np.std(Rs),
        'std R_m1': np.std(Rm1),
        'std R_m2': np.std(Rm2),
    }
    
    return out_dict

def compute_classification(z_np, y_np):
    z_np = z_np.detach().cpu().numpy()
    y_np = y_np.astype(int)
    # Check if binary classification
    unique_classes = np.unique(y_np)
        
    # Initialize classifier with balanced class weights for robustness
    model = LogisticRegression(
        max_iter=1000, 
        class_weight='balanced',
        solver='liblinear',  # Works well for small datasets
        random_state=42
    )
    
    # Use 5-fold cross-validation for robust performance estimation
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer
    
    scoring = {
                'accuracy': make_scorer(accuracy_score)
            }
    cv_results = cross_validate(
                model, z_np, y_np, 
                cv=5, 
                scoring=scoring,
                return_train_score=False
            )
    return cv_results['test_accuracy'].mean()

def evaluate_validation_loss(model, val_dataloader, device, verbose=False, final=False):
    """Evaluate model on validation set."""
    model.eval()
    val_total_loss = 0
    
    # collect all the model outputs
    h1_ins = []
    h2_ins = []
    h1_outs = []
    h2_outs = []
    all_labels = []
    with torch.no_grad():
        for val_batch in val_dataloader:
            h1, h2, x1, x2, label = val_batch
            all_labels.append(label)
            h1 = F.normalize(h1.float(), dim=1).to(device)
            h2 = F.normalize(h2.float(), dim=1).to(device)
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)

            phis = model([h1, h2])

            h1_ins.append(h1.clone().detach().cpu())
            h2_ins.append(h2.clone().detach().cpu())
            h1_outs.append(phis[0].clone().detach().cpu())
            h2_outs.append(phis[1].clone().detach().cpu())

            z_components = model.decouple(phis, full=True, th=model.pruning_threshold)
            losses_list, _, _, _ = model.compute_stage_losses(h1, h2, z_components)
            val_loss = torch.stack(losses_list).mean()
            val_total_loss += val_loss.item()
    # concatenate all the outputs
    all_labels = np.concatenate(all_labels, axis=0)
    all_accuracies = []
    del h1, h2
    h1 = torch.cat(h1_ins, dim=0).to(device)
    h2 = torch.cat(h2_ins, dim=0).to(device)
    h1_outs = torch.cat(h1_outs, dim=0).to(device)
    h2_outs = torch.cat(h2_outs, dim=0).to(device)
    all_accuracies.append(compute_classification(h1_outs, all_labels[:,0]))
    all_accuracies.append(compute_classification(h2_outs, all_labels[:,0]))
    z_n = model.decouple([h1_outs.to(device), h2_outs.to(device)], full=True, th=model.pruning_threshold)
    all_accuracies.append(compute_classification(z_n[0][0], all_labels[:,0]))
    all_accuracies.append(compute_classification(z_n[1][0], all_labels[:,0]))
    all_accuracies.append(compute_classification(z_n[0][1], all_labels[:,0]))
    all_accuracies.append(compute_classification(z_n[1][1], all_labels[:,0]))
    acc_labels = ['Acc h1_out', 'Acc h2_out', 'Acc Zm1', 'Acc Zm2', 'Acc Zs1', 'Acc Zs2']

    #phi1h2, phi2h1 = evaluate_cross_modal_retrieval(h1_outs, h2_outs, projector=model, device=device, labels=all_labels)
    if final:
        all_recalls = []
        batch_size = min(256, h1.shape[0], h2.shape[0])
        z1 = torch.cat((z_n[0][0], z_n[0][1]), dim=1)
        z2 = torch.cat((z_n[1][0], z_n[1][1]), dim=1)
        #assert h1.shape[0] == h2.shape[0], f"h1 and h2 must have the same number of samples but have {h1.shape[0]} and {h2.shape[0]}"
        #assert h1_outs.shape[0] == h2_outs.shape[0], f"h1_outs and h2_outs must have the same number of samples but have {h1_outs.shape[0]} and {h2_outs.shape[0]}"
        #assert z1.shape[0] == h1.shape[0], f"z1 and h1 must have the same number of samples but have {z1.shape[0]} and {h1.shape[0]}"
        all_recalls.append(evaluate_cross_modal_retrieval(z1, h2, device=device, batch_size=batch_size, similarity_model=SimilarityMLP(z1.shape[1], h2.shape[1])))
        all_recalls.append(evaluate_cross_modal_retrieval(z2, h1, device=device, batch_size=batch_size, similarity_model=SimilarityMLP(z2.shape[1], h1.shape[1])))
        all_recalls.append(evaluate_cross_modal_retrieval(h1_outs, h2, device=device, batch_size=batch_size, similarity_model=SimilarityMLP(h1_outs.shape[1], h2.shape[1])))
        all_recalls.append(evaluate_cross_modal_retrieval(h2_outs, h1, device=device, batch_size=batch_size, similarity_model=SimilarityMLP(h2_outs.shape[1], h1.shape[1])))
        all_recalls.append(evaluate_cross_modal_retrieval(h1, h2, device=device, batch_size=batch_size, similarity_model=SimilarityMLP(h1.shape[1], h2.shape[1])))
        all_recalls.append(evaluate_cross_modal_retrieval(h2, h1, device=device, batch_size=batch_size, similarity_model=SimilarityMLP(h2.shape[1], h1.shape[1])))
        recall_labels = ['Recall Z1-Z2', 'Recall Z2-Z1', 'Recall phi1-h2', 'Recall phi2-h1', 'Recall h1-h2', 'Recall h2-h1']

        predictabilities = []
        predictability_labels = []
        #temp_predictability_labels = ['pred-zs1', 'pred-zs2', 'pred-zm1', 'pred-zm2']
        #for label_idx in range(all_labels.shape[1]):
        #    temp_predictability = evaluate_predictability((z_n[0][0], z_n[0][1], z_n[1][0], z_n[1][1]), all_labels, label_idx)
        #    predictabilities.extend(temp_predictability)
        #    predictability_labels.extend([f'{temp_predictability_labels[i]}-{label_idx}' for i in range(len(temp_predictability))])
        regression_df = evaluate_regression(z_n, h1, h2)
        for i in range(len(regression_df)):
            predictabilities.append(regression_df.iloc[i]['r2_mean'])
            predictability_labels.append('Pred ' + regression_df.iloc[i]['name'])

        model.train()
        return val_total_loss / len(val_dataloader), all_accuracies + predictabilities + all_recalls, acc_labels + predictability_labels + recall_labels
    else:
        model.train()
        return val_total_loss / len(val_dataloader), all_accuracies, acc_labels

###
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def evaluate_regression(z_n, h1, h2):
    train_size = int(0.8 * len(h1))
    regression_df = pd.DataFrame(columns=["name", "r2_mean", "r2_std"])
    #print(f'Regression predictability on {train_size} samples (validation on {len(h) - train_size} samples)')
    for i in range(2):
        for j in range(3):
            if i == 0:
                h = h1
            else:
                h = h2
            if j == 2:
                z = torch.cat((z_n[i][0], z_n[i][1]), dim=1)
            else:
                z = z_n[i][j]
            
            # perform "parallel" regression with linear NN
            train_loader = DataLoader(torch.cat((h[:train_size], z[:train_size]), dim=1), batch_size=64, shuffle=True)
            val_loader = DataLoader(torch.cat((h[train_size:], z[train_size:]), dim=1), batch_size=64, shuffle=False)
            linear = torch.nn.Linear(z.shape[1], h.shape[1]).to(z.device)
            optimizer = torch.optim.Adam(linear.parameters(), lr=0.001, weight_decay=0)
            loss_fn = torch.nn.MSELoss()
            early_stopping = 10
            val_losses = []
            #reg_pbar = tqdm(range(1000), desc="Training linear regression", leave=True)
            #for epoch in reg_pbar:
            for epoch in range(1000):
                train_loss = 0
                linear.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    pred = linear(batch[:, h.shape[1]:])
                    #print(batch.shape, pred.shape)
                    loss = loss_fn(pred, batch[:, :h.shape[1]])
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                linear.eval()
                val_losses.append(0)
                for batch in val_loader:
                    #optimizer.zero_grad()
                    with torch.no_grad():
                        pred = linear(batch[:, h.shape[1]:])
                        loss = loss_fn(pred, batch[:, :h.shape[1]])
                        val_losses[-1] += loss.item()
                val_losses[-1] /= len(val_loader)
                if epoch > early_stopping and min(val_losses[-early_stopping:]) > min(val_losses):
                    break
                #reg_pbar.set_postfix({"loss": round(train_loss, 4), "val_loss": round(val_losses[-1], 4)})
            
            h_pred = linear(z)
            h_pred = h_pred.detach().cpu().numpy()
            h_mean = h.mean(0).cpu().numpy()
            r_squares = 1 - (((h.cpu().numpy() - h_pred)**2).sum(0) / ((h.cpu().numpy() - h_mean)**2).sum(0))
            if j == 0:
                name = "Zm"
            elif j == 2:
                name = "(Zm+Zs)"
            else:
                name = "Zs"
            #print(f'{name}{i+1} -----Goodness of fit (R2 score): {r_squares.mean():.3f} (var: {r_squares.std():.3f})')
            regression_df = pd.concat([regression_df, pd.DataFrame({
                "name": [f"{name}{i+1}"],
                "r2_mean": [r_squares.mean()],
                "r2_std": [r_squares.std()]
            })], ignore_index=True)

    return regression_df

def evaluate_classification(z_n, labels):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels for binary classification
    """
    #print(f'Classification predictability')
    components = [
        ("Zs1", z_n[0][1]),  # Shared representation from modality 1
        ("Zs2", z_n[1][1]),  # Shared representation from modality 2
        ("Zm1", z_n[0][0]),  # Modality-specific representation from modality 1
        ("Zm2", z_n[1][0])   # Modality-specific representation from modality 2
    ]

    class_df = pd.DataFrame(columns=["name", "accuracy", "precision", "recall", "f1", "roc_auc"])
    
    # train a classifier on the shared and modality-specific representations
    #print('Task: Binary classification')
    for name, z in components:
        #try:
        # Prepare data
        z_np = z.detach().cpu().numpy()
        y_np = labels.astype(int)
        
        # Check if binary classification
        unique_classes = np.unique(y_np)
        if len(unique_classes) != 2:
            print(f"Warning: Expected binary classification but found {len(unique_classes)} classes. Skipping {name}.")
            continue
            
        # Initialize classifier with balanced class weights for robustness
        model = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            solver='liblinear',  # Works well for small datasets
            random_state=0
        )
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        cv_results = cross_validate(
            model, z_np, y_np, 
            cv=5, 
            scoring=scoring,
            return_train_score=False
        )
        
        # Print results
        #print(f"{name} -----Accuracy:  {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}",
        #      f"  Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}",
        #      f"  Recall:    {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}",
        #      f"  F1 Score:  {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}",
        #      f"  ROC AUC:   {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
        class_df = pd.concat([class_df, pd.DataFrame({
            "name": [name],
            "accuracy": [cv_results['test_accuracy'].mean()],
            "precision": [cv_results['test_precision'].mean()],
            "recall": [cv_results['test_recall'].mean()],
            "f1": [cv_results['test_f1'].mean()],
            "roc_auc": [cv_results['test_roc_auc'].mean()]
        })], ignore_index=True)
        
        #except Exception as e:
        #print(f"Error evaluating {name}: {str(e)}")
        #continue
    
    return class_df

def calc_correlation_matrix(model, threshold=0.05):
    # Get matrices above threshold
    shared_sv = torch.where(torch.linalg.svdvals(model.R_s) > threshold)
    Rs = model.R_s[shared_sv].detach().cpu().numpy()
    m1_sv = torch.where(torch.linalg.svdvals(model.R_m1) > threshold)
    Rm1 = model.R_m1[m1_sv].detach().cpu().numpy()
    m2_sv = torch.where(torch.linalg.svdvals(model.R_m2) > threshold)
    Rm2 = model.R_m2[m2_sv].detach().cpu().numpy()
    matrices = [Rs, Rm1, Rm2]

    names = ['R_s', 'R_m1', 'R_m2']
    corr_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            flat_i = matrices[i].flatten()
            flat_j = matrices[j].flatten()
            if len(flat_i) > len(flat_j):
                flat_j = np.pad(flat_j, (0, len(flat_i) - len(flat_j)))
            elif len(flat_j) > len(flat_i):
                flat_i = np.pad(flat_i, (0, len(flat_j) - len(flat_i)))
            corr_matrix[i,j] = np.corrcoef(flat_i, flat_j)[0,1]
    
    corr_df = pd.DataFrame({
        "name": ['R_s-R_m1', 'R_s-R_m2', 'R_m1-R_m2'],
        "metric": ["correlation"]*3,
        "value": [corr_matrix[0,1], corr_matrix[0,2], corr_matrix[1,2]]
    })
    
    return corr_df

def eval_model(model, test_h1, test_h2, test_labels, device, threshold=0.05):
    h1 = F.normalize(torch.Tensor(test_h1).float(), dim=1).to(device)
    h2 = F.normalize(torch.Tensor(test_h2).float(), dim=1).to(device)
    #h1 = torch.Tensor(test_h1).float().to(device)
    #h2 = torch.Tensor(test_h2).float().to(device)
    phis = model([h1,h2])
    z_n = model.decouple(phis, full=True, th=0.05)
    
    # I would not use the sum as the labels for regression but the use (z_s, z_m) with a linear layer to predict h1
    regression_df = evaluate_regression(z_n, h1, h2)
    # also not entirely sure if the classification worked, since zs gets higher scores than zm
    classification_df = evaluate_classification(z_n, test_labels[:, 0])

    return regression_df, classification_df

def reeval_model(model, test_h1, test_h2, test_labels, device, threshold=0.05):
    corr_df = calc_correlation_matrix(model, threshold)

    #h1 = F.normalize(torch.Tensor(test_h1).float(), dim=1).to(device)
    #h2 = F.normalize(torch.Tensor(test_h2).float(), dim=1).to(device)
    #phis = model([h1,h2])
    #z_n = model.decouple(phis, full=True, th=0.05)
    shared_sv = len(torch.where(torch.linalg.svdvals(model.R_s) > threshold)[0].tolist())
    m1_sv = len(torch.where(torch.linalg.svdvals(model.R_m1) > threshold)[0].tolist())
    m2_sv = len(torch.where(torch.linalg.svdvals(model.R_m2) > threshold)[0].tolist())
    rank_df = pd.DataFrame({
        "name": ["R_s", "R_m1", "R_m2"],
        "metric": ["rank"]*3,
        "value": [shared_sv, m1_sv, m2_sv]
    })
    out_df = pd.concat([corr_df, rank_df], axis=0, ignore_index=True)
    return out_df

###

'''
def evaluate_cross_modal_retrieval(phis0, phis1, projector, device, labels):
    """
    Evaluates cross-modal retrieval performance using cosine similarity.

    Assumes:
    - dataloader yields (h1, h2, x1, x2, l)
    - projector(h1, h2) -> (phi1, phi2)
    - h1: features from modality 1 (e.g., image)
    - h2: features from modality 2 (e.g., text)

    Returns:
        dict with Recall@1, Recall@5, Recall@10 for both directions
    """

    # Compute similarity matrix
    sim_matrix = phis0 @ phis1.T  # (N x N)

    def recall_at_k(sim_matrix, k, labels):
        topk = sim_matrix.topk(k, dim=1).indices
        # Ensure labels is a tensor
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, device=sim_matrix.device)
        # Ensure labels is (batch_size, 1) for broadcasting
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        correct = (topk == labels).any(dim=1).float()
        return correct.mean().item()
    
    k = max(phis0.shape[1], phis1.shape[1])

    ab = recall_at_k(sim_matrix, k, labels[:,2])
    ba = recall_at_k(sim_matrix.T, k, labels[:,1])

    return ab, ba
'''

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
class SklearnTrainer:
    def __init__(self, model, task_type="regression"):
        """
        Wrapper for training a Scikit-Learn model with k-fold cross-validation.

        Args:
            model: A Scikit-Learn model instance (e.g., LinearRegression, RandomForestClassifier).
            task_type (str): "regression" or "classification".
        """
        self.model = model
        self.task_type = task_type

    def train_and_evaluate(self, X, y, k=5):
        """
        Trains and evaluates the model using k-fold cross-validation.

        Args:
            X (numpy array): Input features (N, d).
            y (numpy array): Labels (N,).
            k (int): Number of folds for cross-validation.

        Returns:
            float: Mean validation score (MSE for regression, accuracy for classification).
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        all_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Standardize features
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
            # Train model
            self.model.fit(X_train, y_train)
            # Predict on validation set
            y_pred = self.model.predict(X_val)
            # Compute validation score
            if self.task_type == "regression":
                score = mean_squared_error(y_val, y_pred)  # MSE for regression
            else:
                y_pred = (y_pred > 0.5).astype(int)  # Convert to binary for classification
                score = accuracy_score(y_val, y_pred)  # Accuracy for classification
            all_scores.append(score)

        mean_score = np.mean(all_scores)
        return mean_score, np.var(all_scores)

def evaluate_predictability(z_n, labels, label_idx):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels
        label_idx: Index of the label to evaluate
    """
    #print(f'Predictability for label {label_idx}')
    components = [
        ("Zs1", z_n[1]),  # Shared representation from modality 1
        ("Zs2", z_n[3]),  # Shared representation from modality 2
        ("Zm1", z_n[0]),  # Modality-specific representation from modality 1
        ("Zm2", z_n[2])   # Modality-specific representation from modality 2
    ] 
    
    # Determine if this is a classification or regression task
    y = labels[:,label_idx]
    y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.array(y)
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    
    # Handle edge cases
    if n_unique == 1:
        #print(f"Warning: Label {label_idx} has only one unique value. Skipping evaluation.")
        return None
    
    # Determine task type based on both number of unique values and their nature
    is_classification = (n_unique <= 10 and 
                        np.all(np.mod(y, 1) == 0))  # Check if all values are integers
    
    if is_classification:
        #print(f"Task type: Classification ({n_unique} classes)")
        model = LogisticRegression(max_iter=500)
        task_type = "classification"
        metric_name = "accuracy"
    else:
        #print(f"Task type: Regression ({n_unique} unique values)")
        model = LinearRegression()
        task_type = "regression"
        metric_name = "R2 score"
    
    scores = []
    for name, z in components:
        try:
            reg_model = SklearnTrainer(model=model, task_type=task_type)
            score, score_var = reg_model.train_and_evaluate(z.detach().cpu(), y, k=5)
            scores.append(score.item())
            #print(name, f"-----Predictive performance ({metric_name}): {score:.3f} (var: {score_var:.3f})")
        except Exception as e:
            #print(f"Error evaluating {name}: {str(e)}")
            scores.append(None)
    return scores