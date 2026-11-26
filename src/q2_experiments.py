import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.dataset import FastCIFAR10Data, RANDOM_SEED, CIFAR10_CLASSES
import src.models as models
from src.train_utils import train_classifier, evaluate_loss_acc, evaluate_classifier_detailed, get_device

def get_spatial_permutation(seed=RANDOM_SEED):
    """
    Generates a fixed pseudo-random permutation of the 32x32 = 1024 spatial coordinates.
    Applied identically across all 3 color channels to preserve pixel color values while
    completely destroying 2D spatial locality and spatial neighborhood structure.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(32 * 32)
    return perm

def apply_spatial_permutation(images_tensor, perm):
    """
    images_tensor: (N, 3, 32, 32)
    perm: 1D array of length 1024
    """
    N, C, H, W = images_tensor.shape
    # Flatten spatial dims: (N, C, 1024)
    flat = images_tensor.view(N, C, H * W)
    perm_tensor = torch.from_numpy(perm).long()
    permuted_flat = flat[:, :, perm_tensor]
    return permuted_flat.view(N, C, H, W)

def apply_center_occlusion(images_tensor, box_size=12):
    """
    Masks out a center box of size (box_size x box_size) with zeros (normalized mean).
    images_tensor: (N, 3, 32, 32)
    """
    occ = images_tensor.clone()
    start = (32 - box_size) // 2
    end = start + box_size
    occ[:, :, start:end, start:end] = 0.0
    return occ

def apply_spatial_translation(images_tensor, shift_x=4, shift_y=4):
    """
    Translates images by (shift_x, shift_y) with zero padding.
    images_tensor: (N, 3, 32, 32)
    """
    N, C, H, W = images_tensor.shape
    translated = torch.zeros_like(images_tensor)
    src_y_start = max(0, -shift_y)
    src_y_end = min(H, H - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = min(H, H + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end = min(W, W - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(W, W + shift_x)

    translated[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        images_tensor[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
    return translated

def run_q2_experiments(epochs=15, batch_size=256, results_dir="./results"):
    print("=" * 70)
    print("Starting Q2 Experiments: CNN vs MLP under Clean, Spatially Disturbed, and Occlusion Conditions")
    print("=" * 70)

    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    device = get_device()
    d = FastCIFAR10Data.get_data()

    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

    clean_train_x = d["train_imgs_norm"][train_idx]
    clean_train_y = d["train_targets"][train_idx]
    clean_val_x = d["train_imgs_norm"][val_idx]
    clean_val_y = d["train_targets"][val_idx]
    clean_test_x = d["test_imgs_norm"]
    clean_test_y = d["test_targets"]

    clean_train_loader = DataLoader(TensorDataset(clean_train_x, clean_train_y), batch_size=batch_size, shuffle=True)
    clean_val_loader = DataLoader(TensorDataset(clean_val_x, clean_val_y), batch_size=batch_size, shuffle=False)
    clean_test_loader = DataLoader(TensorDataset(clean_test_x, clean_test_y), batch_size=batch_size, shuffle=False)

    q2_results = {
        "models": {
            "mlp": {"name": "DeepMLP", "param_count": models.count_parameters(models.DeepMLP())},
            "cnn": {"name": "ConvClassifier", "param_count": models.count_parameters(models.ConvClassifier())}
        },
        "clean": {},
        "permuted": {},
        "occluded": {},
        "translated": {}
    }

    # --------------------------------------------------------------------------
    # Part (a): Clean CIFAR-10 Training & Evaluation
    # --------------------------------------------------------------------------
    print("\n--- Q2(a): Clean CIFAR-10 Benchmark ---")
    for mname, model_cls in [("mlp", models.DeepMLP), ("cnn", models.ConvClassifier)]:
        torch.manual_seed(RANDOM_SEED)
        model = model_cls()
        save_ckpt = os.path.join(results_dir, "checkpoints", f"q2_clean_{mname}.pth")
        print(f"Training clean {mname} ({q2_results['models'][mname]['param_count']:,} params)...")
        hist = train_classifier(model, clean_train_loader, clean_val_loader, epochs=epochs, lr=1e-3, device=device, save_path=save_ckpt, verbose=True)

        model.load_state_dict(torch.load(save_ckpt, map_location=device))
        eval_res = evaluate_classifier_detailed(model, clean_test_loader, device=device)
        print(f"[Clean {mname.upper()}] Test Acc: {eval_res['test_acc']:.2f}% | Conf Correct: {eval_res['avg_confidence_correct']:.4f} | Conf Incorrect: {eval_res['avg_confidence_incorrect']:.4f}")

        q2_results["clean"][mname] = {
            "test_acc": eval_res["test_acc"],
            "avg_confidence_correct": eval_res["avg_confidence_correct"],
            "avg_confidence_incorrect": eval_res["avg_confidence_incorrect"],
            "best_val_acc": max(hist["val_acc"]),
            "history": hist
        }

    # --------------------------------------------------------------------------
    # Part (b): Spatially Disturbed CIFAR-10 (Fixed Pixel Permutation)
    # --------------------------------------------------------------------------
    print("\n--- Q2(b): Spatially Disturbed CIFAR-10 (Pixel Permutation) ---")
    perm = get_spatial_permutation(seed=RANDOM_SEED)
    perm_train_x = apply_spatial_permutation(clean_train_x, perm)
    perm_val_x = apply_spatial_permutation(clean_val_x, perm)
    perm_test_x = apply_spatial_permutation(clean_test_x, perm)

    perm_train_loader = DataLoader(TensorDataset(perm_train_x, clean_train_y), batch_size=batch_size, shuffle=True)
    perm_val_loader = DataLoader(TensorDataset(perm_val_x, clean_val_y), batch_size=batch_size, shuffle=False)
    perm_test_loader = DataLoader(TensorDataset(perm_test_x, clean_test_y), batch_size=batch_size, shuffle=False)

    for mname, model_cls in [("mlp", models.DeepMLP), ("cnn", models.ConvClassifier)]:
        torch.manual_seed(RANDOM_SEED)
        model = model_cls()
        save_ckpt = os.path.join(results_dir, "checkpoints", f"q2_permuted_{mname}.pth")
        print(f"Retraining {mname} on Spatially Disturbed CIFAR-10...")
        hist = train_classifier(model, perm_train_loader, perm_val_loader, epochs=epochs, lr=1e-3, device=device, save_path=save_ckpt, verbose=True)

        model.load_state_dict(torch.load(save_ckpt, map_location=device))
        eval_res = evaluate_classifier_detailed(model, perm_test_loader, device=device)
        print(f"[Permuted {mname.upper()}] Test Acc: {eval_res['test_acc']:.2f}% | Conf Correct: {eval_res['avg_confidence_correct']:.4f} | Conf Incorrect: {eval_res['avg_confidence_incorrect']:.4f}")

        q2_results["permuted"][mname] = {
            "test_acc": eval_res["test_acc"],
            "avg_confidence_correct": eval_res["avg_confidence_correct"],
            "avg_confidence_incorrect": eval_res["avg_confidence_incorrect"],
            "best_val_acc": max(hist["val_acc"]),
            "history": hist
        }

    # --------------------------------------------------------------------------
    # Part (c): Third Experimental Condition (Center Occlusion & Translation)
    # --------------------------------------------------------------------------
    print("\n--- Q2(c): Third Experimental Condition (Center Occlusion & Translation) ---")
    occ_train_x = apply_center_occlusion(clean_train_x, box_size=12)
    occ_val_x = apply_center_occlusion(clean_val_x, box_size=12)
    occ_test_x = apply_center_occlusion(clean_test_x, box_size=12)

    occ_train_loader = DataLoader(TensorDataset(occ_train_x, clean_train_y), batch_size=batch_size, shuffle=True)
    occ_val_loader = DataLoader(TensorDataset(occ_val_x, clean_val_y), batch_size=batch_size, shuffle=False)
    occ_test_loader = DataLoader(TensorDataset(occ_test_x, clean_test_y), batch_size=batch_size, shuffle=False)

    for mname, model_cls in [("mlp", models.DeepMLP), ("cnn", models.ConvClassifier)]:
        torch.manual_seed(RANDOM_SEED)
        model = model_cls()
        save_ckpt = os.path.join(results_dir, "checkpoints", f"q2_occluded_{mname}.pth")
        print(f"Training {mname} under Center Occlusion Condition...")
        hist = train_classifier(model, occ_train_loader, occ_val_loader, epochs=epochs, lr=1e-3, device=device, save_path=save_ckpt, verbose=True)

        model.load_state_dict(torch.load(save_ckpt, map_location=device))
        eval_res = evaluate_classifier_detailed(model, occ_test_loader, device=device)
        print(f"[Occluded {mname.upper()}] Test Acc: {eval_res['test_acc']:.2f}% | Conf Correct: {eval_res['avg_confidence_correct']:.4f} | Conf Incorrect: {eval_res['avg_confidence_incorrect']:.4f}")

        q2_results["occluded"][mname] = {
            "test_acc": eval_res["test_acc"],
            "avg_confidence_correct": eval_res["avg_confidence_correct"],
            "avg_confidence_incorrect": eval_res["avg_confidence_incorrect"],
            "best_val_acc": max(hist["val_acc"]),
            "history": hist
        }

    # Also evaluate clean models directly under Spatial Translation shift
    print("\n--- Testing Clean Models on Translated Test Images (Robustness) ---")
    trans_test_x = apply_spatial_translation(clean_test_x, shift_x=4, shift_y=4)
    trans_test_loader = DataLoader(TensorDataset(trans_test_x, clean_test_y), batch_size=batch_size, shuffle=False)

    for mname, model_cls in [("mlp", models.DeepMLP), ("cnn", models.ConvClassifier)]:
        model = model_cls()
        clean_ckpt = os.path.join(results_dir, "checkpoints", f"q2_clean_{mname}.pth")
        model.load_state_dict(torch.load(clean_ckpt, map_location=device))
        eval_res = evaluate_classifier_detailed(model, trans_test_loader, device=device)
        print(f"[Clean {mname.upper()} on Translated Test] Test Acc: {eval_res['test_acc']:.2f}% | Conf Correct: {eval_res['avg_confidence_correct']:.4f}")
        q2_results["translated"][mname] = {
            "test_acc": eval_res["test_acc"],
            "avg_confidence_correct": eval_res["avg_confidence_correct"],
            "avg_confidence_incorrect": eval_res["avg_confidence_incorrect"]
        }

    with open(os.path.join(results_dir, "logs", "q2_summary.json"), "w") as f:
        json.dump(q2_results, f, indent=2)

    return q2_results

if __name__ == "__main__":
    run_q2_experiments(epochs=15, batch_size=256)
