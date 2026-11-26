import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Optimize PyTorch CPU performance
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    save_path=None,
    verbose=True,
    per_subset_eval_fn=None
):
    if device is None:
        device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    if per_subset_eval_fn is not None:
        history["clean_train_acc"] = []
        history["noisy_train_acc"] = []

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_time = time.time() - t0

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["epoch_time"].append(round(epoch_time, 2))

        clean_acc, noisy_acc = None, None
        if per_subset_eval_fn is not None:
            clean_acc, noisy_acc = per_subset_eval_fn(model, device)
            history["clean_train_acc"].append(round(clean_acc, 2))
            history["noisy_train_acc"].append(round(noisy_acc, 2))

        if val_acc > best_val_acc and save_path is not None:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if verbose:
            msg = f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% ({epoch_time:.1f}s)"
            if per_subset_eval_fn is not None and clean_acc is not None:
                msg += f" | CleanTr: {clean_acc:.2f}% NoisyTr: {noisy_acc:.2f}%"
            print(msg)

    if save_path is not None:
        hist_path = os.path.splitext(save_path)[0] + "_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    return history


def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = (correct / total) * 100.0
    return avg_loss, acc


def evaluate_classifier_detailed(model, loader, device=None):
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    accuracy = float((all_preds == all_targets).mean() * 100.0)
    confidences = np.max(all_probs, axis=1)

    correct_mask = (all_preds == all_targets)
    avg_conf_correct = float(np.mean(confidences[correct_mask])) if correct_mask.any() else 0.0
    avg_conf_incorrect = float(np.mean(confidences[~correct_mask])) if (~correct_mask).any() else 0.0

    return {
        "test_acc": round(accuracy, 2),
        "avg_confidence_correct": round(avg_conf_correct, 4),
        "avg_confidence_incorrect": round(avg_conf_incorrect, 4),
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
        "confidences": confidences
    }


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
    save_path=None,
    verbose=True,
    corruption_fn=None
):
    if device is None:
        device = get_device()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "epoch_time": []
    }
    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_mse = 0.0
        total = 0

        for clean_images, _ in train_loader:
            clean_images = clean_images.to(device)
            if corruption_fn is not None:
                inp_images = corruption_fn(clean_images)
            else:
                inp_images = clean_images

            optimizer.zero_grad()
            reconstructions = model(inp_images)
            loss = criterion(reconstructions, clean_images)
            loss.backward()
            optimizer.step()

            running_mse += loss.item() * clean_images.size(0)
            total += clean_images.size(0)

        scheduler.step()
        epoch_time = time.time() - t0

        train_mse = running_mse / total

        model.eval()
        val_mse_running = 0.0
        val_total = 0
        with torch.no_grad():
            for clean_images, _ in val_loader:
                clean_images = clean_images.to(device)
                if corruption_fn is not None:
                    inp_images = corruption_fn(clean_images)
                else:
                    inp_images = clean_images
                reconstructions = model(inp_images)
                loss = criterion(reconstructions, clean_images)
                val_mse_running += loss.item() * clean_images.size(0)
                val_total += clean_images.size(0)

        val_mse = val_mse_running / val_total

        history["epoch"].append(epoch)
        history["train_mse"].append(round(train_mse, 6))
        history["val_mse"].append(round(val_mse, 6))
        history["epoch_time"].append(round(epoch_time, 2))

        if val_mse < best_val_mse and save_path is not None:
            best_val_mse = val_mse
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if verbose:
            print(f"AE Epoch [{epoch:02d}/{epochs:02d}] Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} ({epoch_time:.1f}s)")

    if save_path is not None:
        hist_path = os.path.splitext(save_path)[0] + "_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    return history


def evaluate_autoencoder_mse(model, loader, device=None, corruption_fn=None):
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for clean_images, _ in loader:
            clean_images = clean_images.to(device)
            inp_images = corruption_fn(clean_images) if corruption_fn is not None else clean_images
            reconstructions = model(inp_images)
            loss = criterion(reconstructions, clean_images)
            total_mse += loss.item() * clean_images.size(0)
            total_samples += clean_images.size(0)

    avg_mse = total_mse / total_samples
    return avg_mse
