import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.dataset import get_cifar10_dataloaders, FastCIFAR10Data, RANDOM_SEED
import src.models as models
from src.train_utils import train_classifier, evaluate_loss_acc, evaluate_classifier_detailed, get_device

def run_q1_comparisons(epochs=15, batch_size=256, results_dir="./results"):
    """
    Q1(a): Compares three models under matched parameter (~510k-521k) and training budget:
      1. Deep Global MLP (3072 -> 160 -> 128 -> 64 -> 10, params: 521,898)
      2. Shallow Wide MLP (3072 -> 168 -> 10, params: 518,290)
      3. Locally-Connected Patch MLP (16 patches x 8x8, no weight sharing, params: 510,790)
    """
    print("=" * 70)
    print("Starting Q1(a): Fair MLP Architecture Comparisons")
    print("=" * 70)
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    device = get_device()
    print(f"Using device: {device}")

    comparison_specs = [
        ("deep_mlp", models.DeepMLP, "Deep Global MLP (4 layers, BN, ReLU)"),
        ("shallow_wide_mlp", models.ShallowWideMLP, "Shallow Wide MLP (2 layers, BN, ReLU)"),
        ("lcn_patch_mlp", models.LocallyConnectedPatchMLP, "Locally Connected Patch MLP (16 patches, no weight sharing)")
    ]

    q1_results = {}

    for name, model_cls, desc in comparison_specs:
        print(f"\n--- Training {name}: {desc} ---")
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        model = model_cls()
        param_count = models.count_parameters(model)
        print(f"Trainable parameters: {param_count:,}")

        save_ckpt = os.path.join(results_dir, "checkpoints", f"q1_{name}.pth")
        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            device=device,
            save_path=save_ckpt,
            verbose=True
        )

        model.load_state_dict(torch.load(save_ckpt, map_location=device))
        test_eval = evaluate_classifier_detailed(model, test_loader, device=device)
        print(f"[{name}] Final Official Test Accuracy: {test_eval['test_acc']:.2f}%")

        q1_results[name] = {
            "name": name,
            "description": desc,
            "param_count": param_count,
            "best_val_acc": max(history["val_acc"]),
            "final_val_acc": history["val_acc"][-1],
            "test_acc": test_eval["test_acc"],
            "avg_confidence_correct": test_eval["avg_confidence_correct"],
            "avg_confidence_incorrect": test_eval["avg_confidence_incorrect"],
            "history": history
        }

    with open(os.path.join(results_dir, "logs", "q1_comparisons_summary.json"), "w") as f:
        json.dump(q1_results, f, indent=2)

    return q1_results


def run_q1_memorisation(epochs=20, corruption_ratio=0.4, batch_size=256, results_dir="./results"):
    """
    Q1(b): Deep MLP on 40% label corruption.
    Monitors epoch-by-epoch:
      - Clean training set accuracy
      - Corrupted training set accuracy
      - Official Validation set accuracy (pure clean)
    Demonstrates whether useful structure is learned before memorisation begins.
    """
    print("\n" + "=" * 70)
    print(f"Starting Q1(b): Memorisation Experiment ({corruption_ratio*100:.0f}% Random Label Noise)")
    print("=" * 70)

    d = FastCIFAR10Data.get_data()
    train_x = d["train_imgs_norm"][d["train_idx"]]
    clean_targets = d["train_targets"][d["train_idx"]].clone()
    val_x = d["train_imgs_norm"][d["val_idx"]]
    val_y = d["train_targets"][d["val_idx"]]
    test_x = d["test_imgs_norm"]
    test_y = d["test_targets"]

    n = len(train_x)
    n_corrupted = int(n * corruption_ratio)
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(n)
    corrupted_indices = set(perm[:n_corrupted])

    noisy_targets = clean_targets.clone()
    is_corrupted = np.zeros(n, dtype=bool)

    for i in range(n):
        if i in corrupted_indices:
            orig = int(clean_targets[i].item())
            choices = [c for c in range(10) if c != orig]
            noisy_targets[i] = rng.choice(choices)
            is_corrupted[i] = True

    clean_mask = ~is_corrupted
    corrupted_mask = is_corrupted

    train_loader = DataLoader(TensorDataset(train_x, noisy_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    clean_train_loader = DataLoader(TensorDataset(train_x[clean_mask], clean_targets[clean_mask]), batch_size=batch_size, shuffle=False)
    corrupted_train_loader = DataLoader(TensorDataset(train_x[corrupted_mask], noisy_targets[corrupted_mask]), batch_size=batch_size, shuffle=False)

    device = get_device()
    torch.manual_seed(RANDOM_SEED)
    model = models.DeepMLP().to(device)
    param_count = models.count_parameters(model)
    print(f"Model: DeepMLP | Parameters: {param_count:,} | Corrupted ratio: {corruption_ratio:.2f}")

    def eval_subsets(net, dev):
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        _, clean_acc = evaluate_loss_acc(net, clean_train_loader, criterion, dev)
        _, noisy_acc = evaluate_loss_acc(net, corrupted_train_loader, criterion, dev)
        return clean_acc, noisy_acc

    save_ckpt = os.path.join(results_dir, "checkpoints", "q1_memorisation_mlp.pth")
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device,
        save_path=save_ckpt,
        verbose=True,
        per_subset_eval_fn=eval_subsets
    )

    model.eval()
    test_eval = evaluate_classifier_detailed(model, test_loader, device=device)
    print(f"[Memorisation MLP] Final Official Test Accuracy: {test_eval['test_acc']:.2f}%")

    memo_summary = {
        "corruption_ratio": corruption_ratio,
        "param_count": param_count,
        "clean_train_samples": int(clean_mask.sum()),
        "corrupted_train_samples": int(corrupted_mask.sum()),
        "test_acc": test_eval["test_acc"],
        "history": history
    }

    with open(os.path.join(results_dir, "logs", "q1_memorisation_summary.json"), "w") as f:
        json.dump(memo_summary, f, indent=2)

    return memo_summary


if __name__ == "__main__":
    q1_comp = run_q1_comparisons(epochs=15, batch_size=256)
    q1_memo = run_q1_memorisation(epochs=20, corruption_ratio=0.4, batch_size=256)
    print("\nQ1 Experiments Finished Successfully!")
