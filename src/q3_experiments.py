import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.dataset import FastCIFAR10Data, RANDOM_SEED
import src.models as models
from src.train_utils import train_autoencoder, evaluate_autoencoder_mse, evaluate_classifier_detailed, train_classifier, get_device

def add_gaussian_noise(images, sigma=0.25):
    noise = torch.randn_like(images) * sigma
    return images + noise

def run_q3_experiments(epochs=15, batch_size=256, results_dir="./results"):
    print("=" * 70)
    print("Starting Q3 Experiments: Autoencoders, Denoising, and Representation Probing")
    print("=" * 70)

    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    device = get_device()
    d = FastCIFAR10Data.get_data()

    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

    clean_train_x = d["train_imgs_norm"][train_idx]
    clean_val_x = d["train_imgs_norm"][val_idx]
    clean_test_x = d["test_imgs_norm"]

    clean_train_y = d["train_targets"][train_idx]
    clean_val_y = d["train_targets"][val_idx]
    clean_test_y = d["test_targets"]

    train_loader = DataLoader(TensorDataset(clean_train_x, clean_train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(clean_val_x, clean_val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(clean_test_x, clean_test_y), batch_size=batch_size, shuffle=False)

    q3_results = {
        "part_a": {},
        "part_b": {},
        "part_c": {}
    }

    bottleneck_sizes = [64, 256, 1024]
    trained_aes = {}

    print("\n--- Q3(a): 3 Autoencoders with Bottlenecks (64, 256, 1024) ---")
    for b_size in bottleneck_sizes:
        torch.manual_seed(RANDOM_SEED)
        ae = models.ConvAutoencoder(bottleneck_dim=b_size)
        param_count = models.count_parameters(ae)
        save_ckpt = os.path.join(results_dir, "checkpoints", f"q3_ae_b{b_size}.pth")
        hist_path = os.path.join(results_dir, "checkpoints", f"q3_ae_b{b_size}_history.json")

        if os.path.exists(save_ckpt) and os.path.exists(hist_path):
            print(f"Loading cached AE checkpoint for bottleneck {b_size}...")
            ae.load_state_dict(torch.load(save_ckpt, map_location=device))
            with open(hist_path, "r") as f:
                hist = json.load(f)
        else:
            print(f"Training AE with bottleneck {b_size} ({param_count:,} params)...")
            hist = train_autoencoder(ae, train_loader, val_loader, epochs=epochs, lr=1e-3, device=device, save_path=save_ckpt, verbose=True)

        val_mse = evaluate_autoencoder_mse(ae, val_loader, device=device)
        test_mse = evaluate_autoencoder_mse(ae, test_loader, device=device)
        print(f"[AE Bottleneck {b_size}] Val MSE: {val_mse:.6f} | Test MSE: {test_mse:.6f}")

        trained_aes[b_size] = ae
        q3_results["part_a"][f"ae_{b_size}"] = {
            "bottleneck_size": b_size,
            "param_count": param_count,
            "val_mse": round(float(val_mse), 6),
            "test_mse": round(float(test_mse), 6),
            "history": hist
        }

    best_clean_b = min(bottleneck_sizes, key=lambda b: q3_results["part_a"][f"ae_{b}"]["val_mse"])
    print(f"\n[Q3(a) Selection] Selected clean autoencoder based on validation MSE: Bottleneck {best_clean_b} (Val MSE: {q3_results['part_a'][f'ae_{best_clean_b}']['val_mse']:.6f}, Test MSE: {q3_results['part_a'][f'ae_{best_clean_b}']['test_mse']:.6f})")

    # --------------------------------------------------------------------------
    # Part (b): Denoising Autoencoder & Downstream Classification Recovery
    # --------------------------------------------------------------------------
    print("\n--- Q3(b): Denoising Autoencoder (DAE) with Gaussian Noise (sigma=0.25) ---")
    noise_sigma = 0.25
    corruption_fn = lambda x: add_gaussian_noise(x, sigma=noise_sigma)

    torch.manual_seed(RANDOM_SEED)
    dae = models.ConvAutoencoder(bottleneck_dim=256)
    dae_ckpt = os.path.join(results_dir, "checkpoints", "q3_dae_b256.pth")
    dae_hist_path = os.path.join(results_dir, "checkpoints", "q3_dae_b256_history.json")

    if os.path.exists(dae_ckpt) and os.path.exists(dae_hist_path):
        print("Loading cached DAE checkpoint...")
        dae.load_state_dict(torch.load(dae_ckpt, map_location=device))
        with open(dae_hist_path, "r") as f:
            dae_hist = json.load(f)
    else:
        print("Training DAE...")
        dae_hist = train_autoencoder(dae, train_loader, val_loader, epochs=epochs, lr=1e-3, device=device, save_path=dae_ckpt, verbose=True, corruption_fn=corruption_fn)

    val_rec_mse = evaluate_autoencoder_mse(dae, val_loader, device=device, corruption_fn=corruption_fn)
    test_rec_mse = evaluate_autoencoder_mse(dae, test_loader, device=device, corruption_fn=corruption_fn)
    print(f"[DAE-256] Val Reconstruction MSE: {val_rec_mse:.6f} | Test Reconstruction MSE: {test_rec_mse:.6f}")

    cnn_ckpt = os.path.join(results_dir, "checkpoints", "q2_clean_cnn.pth")
    fixed_cnn = models.ConvClassifier().to(device)
    if os.path.exists(cnn_ckpt):
        fixed_cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device))
        print("Loaded fixed CNN classifier from Q2.")
    else:
        print("Warning: Q2 clean CNN checkpoint not found.")

    # 1. Corrupted test images
    torch.manual_seed(RANDOM_SEED)
    corrupted_test_x = corruption_fn(clean_test_x)
    corrupted_test_loader = DataLoader(TensorDataset(corrupted_test_x, clean_test_y), batch_size=batch_size, shuffle=False)
    eval_corr = evaluate_classifier_detailed(fixed_cnn, corrupted_test_loader, device=device)
    print(f"[Fixed CNN on Corrupted Test Images] Accuracy: {eval_corr['test_acc']:.2f}%")

    # 2. Reconstructions from Clean AE-256
    clean_ae_256 = trained_aes[256]
    clean_ae_256.eval()
    clean_ae_256.to(device)
    with torch.no_grad():
        rec_clean_ae = clean_ae_256(corrupted_test_x.to(device)).cpu()
    rec_clean_loader = DataLoader(TensorDataset(rec_clean_ae, clean_test_y), batch_size=batch_size, shuffle=False)
    eval_clean_rec = evaluate_classifier_detailed(fixed_cnn, rec_clean_loader, device=device)
    print(f"[Fixed CNN on Clean AE-256 Reconstructions] Accuracy: {eval_clean_rec['test_acc']:.2f}%")

    # 3. Reconstructions from DAE-256
    dae.eval()
    dae.to(device)
    with torch.no_grad():
        rec_dae = dae(corrupted_test_x.to(device)).cpu()
    rec_dae_loader = DataLoader(TensorDataset(rec_dae, clean_test_y), batch_size=batch_size, shuffle=False)
    eval_dae_rec = evaluate_classifier_detailed(fixed_cnn, rec_dae_loader, device=device)
    print(f"[Fixed CNN on Denoising AE-256 Reconstructions] Accuracy: {eval_dae_rec['test_acc']:.2f}%")

    q3_results["part_b"] = {
        "corruption": f"Additive Gaussian Noise (mean=0, sigma={noise_sigma})",
        "training_protocol": "Input: corrupted images, Target: clean images, Loss: MSE, Optimizer: AdamW, Scheduler: CosineAnnealingLR",
        "dae_val_mse": round(float(val_rec_mse), 6),
        "dae_test_mse": round(float(test_rec_mse), 6),
        "cnn_on_corrupted_test_acc": eval_corr["test_acc"],
        "cnn_on_clean_ae_rec_test_acc": eval_clean_rec["test_acc"],
        "cnn_on_dae_rec_test_acc": eval_dae_rec["test_acc"],
        "dae_history": dae_hist
    }

    # --------------------------------------------------------------------------
    # Part (c): Linear Probing for Class Information in Learned Representations
    # --------------------------------------------------------------------------
    print("\n--- Q3(c): Probing Class Information vs Reconstruction Quality ---")
    models_to_probe = [
        ("ae_64", trained_aes[64], 64, q3_results["part_a"]["ae_64"]["test_mse"]),
        ("ae_256", trained_aes[256], 256, q3_results["part_a"]["ae_256"]["test_mse"]),
        ("ae_1024", trained_aes[1024], 1024, q3_results["part_a"]["ae_1024"]["test_mse"]),
        ("dae_256", dae, 256, q3_results["part_b"]["dae_test_mse"])
    ]

    for tag, enc_model, feat_dim, test_rec_err in models_to_probe:
        enc_model.eval()
        enc_model.to(device)

        with torch.no_grad():
            feat_train = enc_model.encode(clean_train_x.to(device)).cpu()
            feat_val = enc_model.encode(clean_val_x.to(device)).cpu()
            feat_test = enc_model.encode(clean_test_x.to(device)).cpu()

        probe = models.LinearProbeClassifier(in_features=feat_dim, num_classes=10).to(device)
        probe_train_loader = DataLoader(TensorDataset(feat_train, clean_train_y), batch_size=batch_size, shuffle=True)
        probe_val_loader = DataLoader(TensorDataset(feat_val, clean_val_y), batch_size=batch_size, shuffle=False)
        probe_test_loader = DataLoader(TensorDataset(feat_test, clean_test_y), batch_size=batch_size, shuffle=False)

        save_probe_ckpt = os.path.join(results_dir, "checkpoints", f"q3_probe_{tag}.pth")
        probe_hist = train_classifier(probe, probe_train_loader, probe_val_loader, epochs=15, lr=2e-3, device=device, save_path=save_probe_ckpt, verbose=False)

        probe.load_state_dict(torch.load(save_probe_ckpt, map_location=device))
        probe_eval = evaluate_classifier_detailed(probe, probe_test_loader, device=device)

        print(f"[{tag.upper()}] Test Reconstruction MSE: {test_rec_err:.6f} | Probing Test Classification Acc: {probe_eval['test_acc']:.2f}%")

        q3_results["part_c"][tag] = {
            "feature_dim": feat_dim,
            "test_reconstruction_mse": test_rec_err,
            "probe_val_acc": max(probe_hist["val_acc"]),
            "probe_test_acc": probe_eval["test_acc"]
        }

    with open(os.path.join(results_dir, "logs", "q3_summary.json"), "w") as f:
        json.dump(q3_results, f, indent=2)

    return q3_results

if __name__ == "__main__":
    run_q3_experiments(epochs=15, batch_size=256)
