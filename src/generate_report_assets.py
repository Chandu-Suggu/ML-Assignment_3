import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.dataset import FastCIFAR10Data, RANDOM_SEED, CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
import src.models as models

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def unnormalize(tensor_img):
    """Converts normalized tensor back to (H, W, C) numpy array in [0, 1]."""
    # tensor_img: (3, H, W)
    mean = CIFAR10_MEAN.view(3, 1, 1)
    std = CIFAR10_STD.view(3, 1, 1)
    unnorm = tensor_img.clone() * std + mean
    unnorm = torch.clamp(unnorm, 0.0, 1.0)
    return unnorm.permute(1, 2, 0).cpu().numpy()

def generate_all_figures(results_dir="./results", figures_dir="./results/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    print("=" * 70)
    print("Generating Publication-Quality Figures & Visual Deliverables")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Figure 1: Q1(a) Architecture Comparisons (Deep vs Wide vs LCN)
    # --------------------------------------------------------------------------
    q1_comp = load_json(os.path.join(results_dir, "logs", "q1_comparisons_summary.json"))
    if q1_comp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
        colors = {"deep_mlp": "#1f77b4", "shallow_wide_mlp": "#ff7f0e", "lcn_patch_mlp": "#2ca02c"}
        labels = {
            "deep_mlp": "Deep Global MLP (4 layers, 522k)",
            "shallow_wide_mlp": "Shallow Wide MLP (2 layers, 518k)",
            "lcn_patch_mlp": "Locally Connected Patch MLP (511k)"
        }

        for k, data in q1_comp.items():
            hist = data["history"]
            epochs = hist["epoch"]
            c = colors.get(k, "black")
            lbl = labels.get(k, k)
            ax1.plot(epochs, hist["train_loss"], label=f"{lbl} (Train)", color=c, linestyle="--", alpha=0.7)
            ax1.plot(epochs, hist["val_loss"], label=f"{lbl} (Val)", color=c, linewidth=2.0)

            ax2.plot(epochs, hist["train_acc"], label=f"{lbl} (Train)", color=c, linestyle="--", alpha=0.7)
            ax2.plot(epochs, hist["val_acc"], label=f"{lbl} (Val)", color=c, linewidth=2.0)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.set_title("(a) Training & Validation Loss Trajectories")
        ax1.legend(frameon=True, fontsize=8)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("(b) Training & Validation Accuracy Trajectories")
        ax2.legend(frameon=True, fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig1_q1_architectures.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig1_q1_architectures.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 1: Q1 Architecture Comparison.")

    # --------------------------------------------------------------------------
    # Figure 2: Q1(b) Memorisation Trajectory
    # --------------------------------------------------------------------------
    q1_memo = load_json(os.path.join(results_dir, "logs", "q1_memorisation_summary.json"))
    if q1_memo:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        hist = q1_memo["history"]
        epochs = hist["epoch"]

        ax.plot(epochs, hist["clean_train_acc"], color="#2ca02c", linewidth=2.2, label="Clean Training Subset (60%)", marker="o", markersize=4)
        ax.plot(epochs, hist["noisy_train_acc"], color="#d62728", linewidth=2.2, label="Corrupted Training Subset (40%)", marker="s", markersize=4)
        ax.plot(epochs, hist["val_acc"], color="#1f77b4", linewidth=2.5, label="Clean Official Validation Set", marker="^", markersize=5)

        peak_epoch = int(np.argmax(hist["val_acc"])) + 1
        peak_val = max(hist["val_acc"])
        ax.axvline(x=peak_epoch, color="gray", linestyle=":", alpha=0.8)
        ax.annotate(f"Peak Val Acc: {peak_val:.1f}%\n(Structure Learned)",
                    xy=(peak_epoch, peak_val), xytext=(peak_epoch + 1.2, peak_val - 5),
                    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                    fontsize=9, fontweight="bold")

        ax.annotate("Late Memorisation Phase\n(Noisy Acc Rises, Val Acc Degrades)",
                    xy=(epochs[-1], hist["noisy_train_acc"][-1]), xytext=(epochs[-1] - 8, hist["noisy_train_acc"][-1] - 10),
                    arrowprops=dict(facecolor="red", shrink=0.08, width=1, headwidth=5),
                    fontsize=9, color="#d62728")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Q1(b) Memorisation Dynamics under 40% Random Label Noise")
        ax.set_ylim(0, 100)
        ax.legend(frameon=True, loc="center right")

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig2_q1_memorisation.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig2_q1_memorisation.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 2: Q1 Memorisation Trajectory.")

    # --------------------------------------------------------------------------
    # Figure 3 & 4: Q2 CNN vs MLP across Conditions & Confidence Analysis
    # --------------------------------------------------------------------------
    q2_summary = load_json(os.path.join(results_dir, "logs", "q2_summary.json"))
    if q2_summary:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        conditions = ["Clean", "Spatially Permuted", "Center Occluded", "Translated (4px)"]
        cond_keys = ["clean", "permuted", "occluded", "translated"]

        mlp_accs = [q2_summary[k]["mlp"]["test_acc"] for k in cond_keys]
        cnn_accs = [q2_summary[k]["cnn"]["test_acc"] for k in cond_keys]

        x = np.arange(len(conditions))
        width = 0.35

        rects1 = ax.bar(x - width/2, mlp_accs, width, label="Deep MLP (522k)", color="#4c72b0")
        rects2 = ax.bar(x + width/2, cnn_accs, width, label="ConvClassifier (572k)", color="#c44e52")

        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Q2: CNN vs MLP Performance across Structural Perturbations")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(frameon=True)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig3_q2_comparison.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig3_q2_comparison.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 3: Q2 Comparison across Perturbations.")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        cond_labels = ["Clean", "Permuted", "Occluded"]
        sub_keys = ["clean", "permuted", "occluded"]

        cnn_conf_corr = [q2_summary[k]["cnn"]["avg_confidence_correct"] * 100 for k in sub_keys]
        cnn_conf_inc = [q2_summary[k]["cnn"]["avg_confidence_incorrect"] * 100 for k in sub_keys]
        mlp_conf_corr = [q2_summary[k]["mlp"]["avg_confidence_correct"] * 100 for k in sub_keys]
        mlp_conf_inc = [q2_summary[k]["mlp"]["avg_confidence_incorrect"] * 100 for k in sub_keys]

        x = np.arange(len(cond_labels))
        w = 0.2
        ax.bar(x - 1.5*w, cnn_conf_corr, w, label="CNN (Correct)", color="#2ca02c")
        ax.bar(x - 0.5*w, cnn_conf_inc, w, label="CNN (Incorrect)", color="#98df8a")
        ax.bar(x + 0.5*w, mlp_conf_corr, w, label="MLP (Correct)", color="#1f77b4")
        ax.bar(x + 1.5*w, mlp_conf_inc, w, label="MLP (Incorrect)", color="#aec7e8")

        ax.set_ylabel("Average Softmax Confidence (%)")
        ax.set_title("Q2(c): Confidence Calibration on Correct vs. Incorrect Predictions")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(frameon=True, ncol=2, fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig4_q2_confidence_error.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig4_q2_confidence_error.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 4: Q2 Confidence Analysis.")

    # --------------------------------------------------------------------------
    # Figure 5: Q3 Visual Image Reconstructions (Clean vs Corrupted vs DAE)
    # --------------------------------------------------------------------------
    d = FastCIFAR10Data.get_data()
    clean_test_x = d["test_imgs_norm"]
    clean_test_y = d["test_targets"]

    clean_ae = models.ConvAutoencoder(bottleneck_dim=256)
    dae = models.ConvAutoencoder(bottleneck_dim=256)
    clean_ae_path = os.path.join(results_dir, "checkpoints", "q3_ae_b256.pth")
    dae_path = os.path.join(results_dir, "checkpoints", "q3_dae_b256.pth")

    if os.path.exists(clean_ae_path) and os.path.exists(dae_path):
        clean_ae.load_state_dict(torch.load(clean_ae_path, map_location="cpu"))
        dae.load_state_dict(torch.load(dae_path, map_location="cpu"))
        clean_ae.eval()
        dae.eval()

        # Select 8 sample indices from diverse classes
        sample_indices = []
        for c in range(8):
            idx = int(np.where(clean_test_y.numpy() == c)[0][0])
            sample_indices.append(idx)

        samples_clean = clean_test_x[sample_indices]
        noise = torch.randn_like(samples_clean) * 0.25
        samples_noisy = samples_clean + noise

        with torch.no_grad():
            rec_clean_ae = clean_ae(samples_noisy)
            rec_dae = dae(samples_noisy)

        fig, axes = plt.subplots(4, 8, figsize=(12, 6.5), dpi=300)
        row_titles = ["Clean Input", "Corrupted (Noise)", "Clean AE (b=256)", "DAE (b=256)"]

        for i in range(8):
            cls_name = CIFAR10_CLASSES[i].capitalize()
            axes[0, i].imshow(unnormalize(samples_clean[i]))
            axes[0, i].set_title(cls_name, fontsize=9, fontweight="bold")
            axes[0, i].axis("off")

            axes[1, i].imshow(unnormalize(samples_noisy[i]))
            axes[1, i].axis("off")

            axes[2, i].imshow(unnormalize(rec_clean_ae[i]))
            axes[2, i].axis("off")

            axes[3, i].imshow(unnormalize(rec_dae[i]))
            axes[3, i].axis("off")

        for r in range(4):
            axes[r, 0].text(-0.35, 0.5, row_titles[r], transform=axes[r, 0].transAxes,
                            rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")

        plt.suptitle("Q3(b): Visual Reconstructions across Clean, Corrupted, and Autoencoders", fontsize=12, y=0.98)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig5_q3_ae_reconstruction.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig5_q3_ae_reconstruction.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 5: Q3 Visual Reconstructions.")

    # --------------------------------------------------------------------------
    # Figure 6: Q3 Class Information vs Reconstruction Error
    # --------------------------------------------------------------------------
    q3_summary = load_json(os.path.join(results_dir, "logs", "q3_summary.json"))
    if q3_summary:
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        models_q3 = ["ae_64", "ae_256", "ae_1024", "dae_256"]
        names = ["AE (b=64)", "AE (b=256)", "AE (b=1024)", "DAE (b=256)"]

        mse_vals = [q3_summary["part_c"][m]["test_reconstruction_mse"] for m in models_q3]
        probe_accs = [q3_summary["part_c"][m]["probe_test_acc"] for m in models_q3]

        color = "#1f77b4"
        ax1.set_xlabel("Autoencoder Architecture")
        ax1.set_ylabel("Test Reconstruction MSE (Lower is Better)", color=color)
        b1 = ax1.bar(np.arange(len(names)) - 0.2, mse_vals, 0.4, color=color, alpha=0.85, label="Reconstruction MSE")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = "#d62728"
        ax2.set_ylabel("Probed Test Classification Acc (%) (Higher is Better)", color=color)
        b2 = ax2.bar(np.arange(len(names)) + 0.2, probe_accs, 0.4, color=color, alpha=0.85, label="Linear Probe Accuracy")
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.set_xticks(np.arange(len(names)))
        ax1.set_xticklabels(names, fontweight="bold")
        ax1.set_title("Q3(c): Reconstruction Quality vs. Semantic Class Information")

        for rect in b1:
            h = rect.get_height()
            ax1.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#1f77b4")
        for rect in b2:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#d62728")

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig6_q3_class_vs_reconstruction.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig6_q3_class_vs_reconstruction.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 6: Q3 Class Info vs Reconstruction MSE.")

    # --------------------------------------------------------------------------
    # Figure 7: Q4 SVM Hyperparameter Sweep Curves
    # --------------------------------------------------------------------------
    q4_summary = load_json(os.path.join(results_dir, "logs", "q4_summary.json"))
    if q4_summary:
        sweep = q4_summary["part_a"]["C_kernel_sweep"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

        kernels = ["linear", "poly", "rbf"]
        colors = {"linear": "#1f77b4", "poly": "#ff7f0e", "rbf": "#2ca02c"}
        markers = {"linear": "o", "poly": "s", "rbf": "^"}

        for k in kernels:
            k_entries = [e for e in sweep if e["kernel"] == k]
            c_vals = [e["C"] for e in k_entries]
            val_accs = [e["val_acc"] for e in k_entries]
            sv_fracs = [e["sv_fraction"] for e in k_entries]

            ax1.plot(c_vals, val_accs, label=f"Kernel: {k.upper()}", color=colors[k], marker=markers[k], linewidth=2.0)
            ax2.plot(c_vals, sv_fracs, label=f"Kernel: {k.upper()}", color=colors[k], marker=markers[k], linewidth=2.0)

        ax1.set_xscale("log")
        ax1.set_xlabel("Penalty Parameter C")
        ax1.set_ylabel("Validation Accuracy (%)")
        ax1.set_title("(a) Validation Accuracy vs. C")
        ax1.legend(frameon=True)

        ax2.set_xscale("log")
        ax2.set_xlabel("Penalty Parameter C")
        ax2.set_ylabel("Support Vector Fraction")
        ax2.set_title("(b) Support Vector Fraction vs. C")
        ax2.legend(frameon=True)

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig7_q4_svm_sweep.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig7_q4_svm_sweep.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 7: Q4 SVM Parameter Sweep.")

        # ----------------------------------------------------------------------
        # Figure 8: Q4 Boundary Analysis on Difficult Class Pairs
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.5), dpi=300)
        diff_pairs = q4_summary["part_b"]["difficult_pairs"]

        for row_idx, pair_data in enumerate(diff_pairs):
            c1_name = pair_data["class1"]
            c2_name = pair_data["class2"]
            top_svs = pair_data["top_support_vectors"]
            near_boundary = pair_data["closest_test_examples"]

            # Show 2 support vectors
            for sv_col in range(2):
                sv_info = top_svs[sv_col]
                tr_idx = sv_info["train_index"]
                img_t = d["train_imgs_norm"][d["train_idx"]][tr_idx]
                ax = axes[row_idx, sv_col]
                ax.imshow(unnormalize(img_t))
                ax.set_title(f"SV ({sv_info['label']})\nα={sv_info['dual_coefficient']:.2f}", fontsize=8)
                ax.axis("off")

            # Show 2 near-boundary test samples
            for te_col in range(2):
                te_info = near_boundary[te_col]
                te_idx = te_info["test_index"]
                img_t = d["test_imgs_norm"][te_idx]
                ax = axes[row_idx, 2 + te_col]
                ax.imshow(unnormalize(img_t))
                status = "ERR" if te_info["is_error"] else "OK"
                ax.set_title(f"Test ({te_info['true_label']})\nPred:{te_info['predicted_label']} f={te_info['decision_score']:.2f} [{status}]", fontsize=8)
                ax.axis("off")

            axes[row_idx, 0].text(-0.35, 0.5, f"{c1_name.capitalize()} vs {c2_name.capitalize()}\n({pair_data['mutual_test_errors']} errors)",
                                   transform=axes[row_idx, 0].transAxes, rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")

        plt.suptitle("Q4(b): Support Vectors and Near-Boundary Test Images for Difficult Pairs", fontsize=12, y=0.98)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig8_q4_boundary_support_vectors.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig8_q4_boundary_support_vectors.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 8: Q4 Boundary Support Vectors.")

        # ----------------------------------------------------------------------
        # Figure 9: Q4 Contingency Heatmap
        # ----------------------------------------------------------------------
        contingency = np.array(q4_summary["part_c"]["contingency_table"])
        fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
        im = ax.imshow(contingency, cmap="Blues", aspect="auto")

        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels([f"Cluster {i}" for i in range(10)], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([c.capitalize() for c in CIFAR10_CLASSES], fontsize=9)
        ax.set_xlabel("Assigned Cluster (Unsupervised K-Means)", fontweight="bold")
        ax.set_ylabel("True Ground-Truth Class", fontweight="bold")
        ax.set_title(f"Q4(c) Test Cluster-to-Class Contingency Table\n(ARI: {q4_summary['part_c']['test_ari']:.4f}, NMI: {q4_summary['part_c']['test_nmi']:.4f})")

        for i in range(10):
            for j in range(10):
                val = contingency[i, j]
                color = "white" if val > contingency.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig9_q4_contingency_heatmap.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(figures_dir, "fig9_q4_contingency_heatmap.png"), bbox_inches="tight")
        plt.close(fig)
        print("Generated Fig 9: Q4 Contingency Heatmap.")

    print("\nAll figures generated successfully in:", figures_dir)

if __name__ == "__main__":
    generate_all_figures()
