import os
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

STUDENT_NAME = "Suggu Chandu"
ROLL_NUMBER = "M26CSA017"
RANDOM_SEED = 26017

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

def get_or_create_stratified_split(data_root="./data", split_save_path="./data/split_indices.npz"):
    """
    Creates a stratified 45,000 train / 5,000 validation split from the 50,000 CIFAR-10 training images
    using random seed 26017 (numeric characters of roll number M26CSA017 in order).
    """
    if os.path.exists(split_save_path):
        data = np.load(split_save_path)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        return train_idx, val_idx

    raw_trainset = datasets.CIFAR10(root=data_root, train=True, download=False)
    targets = np.array(raw_trainset.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, train_size=45000, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    os.makedirs(os.path.dirname(split_save_path), exist_ok=True)
    np.savez_compressed(split_save_path, train_idx=train_idx, val_idx=val_idx, seed=RANDOM_SEED, roll_number=ROLL_NUMBER)
    print(f"[Split] Created stratified split: {len(train_idx)} train, {len(val_idx)} val using seed {RANDOM_SEED}.")
    return train_idx, val_idx

class FastCIFAR10Data:
    """
    Loads raw CIFAR-10 into memory as normalized PyTorch Tensors for ultra-fast CPU/GPU training.
    """
    _cached_data = None

    @classmethod
    def get_data(cls, data_root="./data"):
        if cls._cached_data is None:
            raw_train = datasets.CIFAR10(root=data_root, train=True, download=False)
            raw_test = datasets.CIFAR10(root=data_root, train=False, download=False)

            train_imgs = torch.from_numpy(raw_train.data).permute(0, 3, 1, 2).float() / 255.0
            test_imgs = torch.from_numpy(raw_test.data).permute(0, 3, 1, 2).float() / 255.0

            # Normalized versions
            train_imgs_norm = (train_imgs - CIFAR10_MEAN) / CIFAR10_STD
            test_imgs_norm = (test_imgs - CIFAR10_MEAN) / CIFAR10_STD

            train_targets = torch.tensor(raw_train.targets, dtype=torch.long)
            test_targets = torch.tensor(raw_test.targets, dtype=torch.long)

            train_idx, val_idx = get_or_create_stratified_split(data_root=data_root)

            cls._cached_data = {
                "train_imgs": train_imgs,
                "test_imgs": test_imgs,
                "train_imgs_norm": train_imgs_norm,
                "test_imgs_norm": test_imgs_norm,
                "train_targets": train_targets,
                "test_targets": test_targets,
                "train_idx": train_idx,
                "val_idx": val_idx
            }
        return cls._cached_data

def get_cifar10_dataloaders(data_root="./data", batch_size=128, normalize=True, sample_transform_fn=None):
    """
    Returns train_loader (45,000), val_loader (5,000), and test_loader (10,000).
    Using in-memory TensorDatasets for max CPU throughput.
    """
    d = FastCIFAR10Data.get_data(data_root=data_root)
    imgs = d["train_imgs_norm"] if normalize else d["train_imgs"]
    test_imgs = d["test_imgs_norm"] if normalize else d["test_imgs"]

    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

    train_x = imgs[train_idx]
    train_y = d["train_targets"][train_idx]
    val_x = imgs[val_idx]
    val_y = d["train_targets"][val_idx]
    test_x = test_imgs
    test_y = d["test_targets"]

    if sample_transform_fn is not None:
        train_x = sample_transform_fn(train_x)
        val_x = sample_transform_fn(val_x)
        test_x = sample_transform_fn(test_x)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
