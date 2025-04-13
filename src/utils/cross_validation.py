import numpy as np
import torch
from sklearn.model_selection import KFold


def get_k_folds(data, k=5, random_state=42):
    """
    Splits dataset indices into K folds.

    Args:
        data (list or dataset): The dataset (list of tensors or PyTorch dataset).
        k (int): Number of folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        Generator of (train_indices, val_indices).
    """
    num_samples = len(data)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(np.arange(num_samples)):
        yield train_idx, val_idx


def create_fold_datasets(dataset, train_idx, val_idx, batch_size=64, num_workers=4):
    """
    Creates DataLoader objects for a specific fold.

    Args:
        dataset: PyTorch dataset
        train_idx: Indices for training set
        val_idx: Indices for validation set
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
