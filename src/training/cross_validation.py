import numpy as np
from sklearn.model_selection import KFold


def get_k_folds(data, k=5):
    """
    Splits dataset indices into K folds.

    Args:
        data (list or dataset): The dataset (list of tensors or PyTorch dataset).
        k (int): Number of folds.

    Returns:
        Generator of (train_indices, val_indices).
    """
    num_samples = len(data)  # Number of images
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(np.arange(num_samples)):
        yield train_idx, val_idx
