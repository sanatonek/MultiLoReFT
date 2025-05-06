import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    components = [
        ("Zs1", z_n[0][1]),  # Shared representation from modality 1
        ("Zs2", z_n[1][1]),  # Shared representation from modality 2
        ("Zm1", z_n[0][0]),  # Modality-specific representation from modality 1
        ("Zm2", z_n[1][0])   # Modality-specific representation from modality 2
    ] 
    
    # Determine if this is a classification or regression task
    y = labels[:,label_idx]
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    
    # Handle edge cases
    if n_unique == 1:
        print(f"Warning: Label {label_idx} has only one unique value. Skipping evaluation.")
        return {}
    
    # Determine task type based on both number of unique values and their nature
    is_classification = (n_unique <= 10 and 
                        np.all(np.mod(y, 1) == 0))  # Check if all values are integers
    
    if is_classification:
        model = LogisticRegression(max_iter=500)
        task_type = "classification"
        metric_name = "accuracy"
    else:
        model = LinearRegression()
        task_type = "regression"
        metric_name = "R2 score"
    
    out_dict = {'component': [], 'metric': [], 'score': [], 'var': []}
    for name, z in components:
        try:
            reg_model = SklearnTrainer(model=model, task_type=task_type)
            score, score_var = reg_model.train_and_evaluate(z.detach().cpu(), y, k=5)
            out_dict['component'].append(name)
            out_dict['metric'].append(metric_name)
            out_dict['score'].append(score)
            out_dict['var'].append(score_var)
        except Exception as e:
            continue
    return out_dict

def evaluate_regression(z, h, i=0):
    train_size = int(0.8 * len(h))
    z = torch.cat(z, dim=1)

    # perform "parallel" regression with linear NN
    train_loader = DataLoader(torch.cat((h[:train_size], z[:train_size]), dim=1), batch_size=64, shuffle=True)
    val_loader = DataLoader(torch.cat((h[train_size:], z[train_size:]), dim=1), batch_size=64, shuffle=False)
    linear = torch.nn.Linear(z.shape[1], h.shape[1]).to(z.device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.0001, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    early_stopping = 10
    val_losses = []
    for epoch in range(500):
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
    
    h_pred = linear(z[train_size:])
    h_pred = h_pred.detach().cpu().numpy()
    h_mean = h[train_size:].mean(0).cpu().numpy()
    r_squares = 1 - (((h[train_size:].cpu().numpy() - h_pred)**2).sum(0) / ((h[train_size:].cpu().numpy() - h_mean)**2).sum(0))
    out_dict = {
        'component': [f'Zs{i}-Zm{i}'],
        'metric': ['R2 score (multiregression)'],
        'score': [r_squares.mean()],
        'var': [r_squares.std()]
    }
    return out_dict

def evaluate_classification(z_n, labels):
    """Evaluate how well each component (shared and modality-specific) can predict the target label.
    
    Args:
        z_n: Tuple of (modality_specific, shared) representations for each modality
        labels: Target labels for binary classification
    """
    components = [
        ("Zs1", z_n[0][1]),  # Shared representation from modality 1
        ("Zs2", z_n[1][1]),  # Shared representation from modality 2
        ("Zm1", z_n[0][0]),  # Modality-specific representation from modality 1
        ("Zm2", z_n[1][0])   # Modality-specific representation from modality 2
    ]
    
    # train a classifier on the shared and modality-specific representations
    out_dict = {'component': [], 'metric': [], 'score': [], 'var': []}
    for name, z in components:
        try:
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
                random_state=42
            )
            
            # Use 5-fold cross-validation for robust performance estimation
            from sklearn.model_selection import cross_validate
            from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
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
            out_dict['component'].extend([name] * 5)
            out_dict['metric'].extend(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
            out_dict['score'].extend([
                cv_results['test_accuracy'].mean(), 
                cv_results['test_precision'].mean(), 
                cv_results['test_recall'].mean(), 
                cv_results['test_f1'].mean(), 
                cv_results['test_roc_auc'].mean()
            ])
            out_dict['var'].extend([
                cv_results['test_accuracy'].std(), 
                cv_results['test_precision'].std(), 
                cv_results['test_recall'].std(), 
                cv_results['test_f1'].std(), 
                cv_results['test_roc_auc'].std()
            ])
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    return out_dict