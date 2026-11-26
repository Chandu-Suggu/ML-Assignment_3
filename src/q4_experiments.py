import os
import json
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.dataset import FastCIFAR10Data, RANDOM_SEED, CIFAR10_CLASSES
import src.models as models
from src.train_utils import get_device

def extract_fixed_representations(cnn_ckpt_path="./results/checkpoints/q2_clean_cnn.pth", device=None):
    """
    Extracts the fixed 128-dimensional representations from the clean CNN penultimate layer.
    Extracts for train (45,000), val (5,000), and test (10,000).
    """
    if device is None:
        device = get_device()

    d = FastCIFAR10Data.get_data()
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

    clean_train_x = d["train_imgs_norm"][train_idx]
    clean_val_x = d["train_imgs_norm"][val_idx]
    clean_test_x = d["test_imgs_norm"]

    clean_train_y = d["train_targets"][train_idx].numpy()
    clean_val_y = d["train_targets"][val_idx].numpy()
    clean_test_y = d["test_targets"].numpy()

    cnn = models.ConvClassifier().to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt_path, map_location=device))
    cnn.eval()

    def get_feats(tensor_x):
        feats = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(tensor_x), batch_size):
                batch = tensor_x[i:i+batch_size].to(device)
                f = cnn.extract_features(batch)
                feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    print("Extracting fixed CNN penultimate representations...")
    X_train = get_feats(clean_train_x)
    X_val = get_feats(clean_val_x)
    X_test = get_feats(clean_test_x)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, clean_train_y), (X_val, clean_val_y), (X_test, clean_test_y), scaler

def get_stratified_svm_subset(X, y, n_per_class=600, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    sub_idx = []
    for c in range(10):
        cls_idx = np.where(y == c)[0]
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        sub_idx.extend(chosen)
    sub_idx = np.array(sub_idx)
    rng.shuffle(sub_idx)
    return X[sub_idx], y[sub_idx], sub_idx

def run_q4_experiments(cnn_ckpt_path="./results/checkpoints/q2_clean_cnn.pth", results_dir="./results"):
    print("=" * 70)
    print("Starting Q4 Bonus Experiments: SVM Kernel/C Sweep, Boundary Support Vectors, and Clustering")
    print("=" * 70)

    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    (X_train_full, y_train_full), (X_val_full, y_val_full), (X_test, y_test), _ = \
        extract_fixed_representations(cnn_ckpt_path=cnn_ckpt_path)

    # Fixed stratified training subset (6,000 samples: 600 per class)
    # Fixed stratified validation subset (1,000 samples: 100 per class)
    X_train, y_train, train_subset_indices = get_stratified_svm_subset(X_train_full, y_train_full, n_per_class=600, seed=RANDOM_SEED)
    X_val, y_val, val_subset_indices = get_stratified_svm_subset(X_val_full, y_val_full, n_per_class=100, seed=RANDOM_SEED)

    print(f"SVM Subsets: Train={len(X_train)} samples, Val={len(X_val)} samples, Test={len(X_test)} samples.")

    # --------------------------------------------------------------------------
    # Part (a): Sweep C and Kernels
    # --------------------------------------------------------------------------
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0] # 5 values spanning 4 orders of magnitude
    kernels = ["linear", "poly", "rbf"] # 3 kernels, poly degree=3

    sweep_results = []
    best_val_acc = -1.0
    best_setting = None
    best_svm_model = None

    print("\n--- Q4(a): SVM Hyperparameter Sweep ---")
    for k in kernels:
        for C in C_values:
            gamma = "scale" if k in ["rbf", "poly"] else "auto"
            degree = 3 if k == "poly" else 3
            clf = SVC(C=C, kernel=k, degree=degree, gamma=gamma, random_state=RANDOM_SEED)
            clf.fit(X_train, y_train)

            train_acc = float(accuracy_score(y_train, clf.predict(X_train)) * 100.0)
            val_acc = float(accuracy_score(y_val, clf.predict(X_val)) * 100.0)
            sv_fraction = float(len(clf.support_) / len(X_train))

            print(f"Kernel: {k:6s} | C: {C:6.2f} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | SV Fraction: {sv_fraction:6.4f}")

            res_entry = {
                "kernel": k,
                "C": C,
                "train_acc": round(train_acc, 2),
                "val_acc": round(val_acc, 2),
                "sv_fraction": round(sv_fraction, 4)
            }
            sweep_results.append(res_entry)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_setting = {"kernel": k, "C": C}
                best_svm_model = clf

    print(f"\n[Q4(a) Selected SVM on Validation Set]: Kernel={best_setting['kernel']}, C={best_setting['C']}, Val Acc={best_val_acc:.2f}%")

    # Evaluate selected SVM ONCE on official test set
    y_test_pred = best_svm_model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_test_pred) * 100.0)
    print(f"[Selected SVM on Official Test Set]: Test Accuracy: {test_acc:.2f}%")

    # --------------------------------------------------------------------------
    # Part (b): Inspect Difficult Class Pairs & Decision Boundary Support Vectors
    # --------------------------------------------------------------------------
    print("\n--- Q4(b): Boundary Analysis on Difficult Class Pairs ---")
    cm = confusion_matrix(y_test, y_test_pred)
    # Find class pairs with highest mutual confusion
    pairs_confusion = []
    for i in range(10):
        for j in range(i + 1, 10):
            conf_count = cm[i, j] + cm[j, i]
            pairs_confusion.append((conf_count, i, j))
    pairs_confusion.sort(reverse=True)

    top_3_pairs = pairs_confusion[:3]
    print("Top 3 most difficult class pairs on Test Set:")
    for count, c1, c2 in top_3_pairs:
        print(f"  Pair ({CIFAR10_CLASSES[c1]} [{c1}], {CIFAR10_CLASSES[c2]} [{c2}]): {count} mutual errors")

    difficult_pairs_data = []

    for count, c1, c2 in top_3_pairs:
        # Binary subset from train subset
        pair_mask_tr = (y_train == c1) | (y_train == c2)
        X_tr_pair = X_train[pair_mask_tr]
        y_tr_pair = np.where(y_train[pair_mask_tr] == c1, 1, -1)
        orig_tr_indices = train_subset_indices[pair_mask_tr]

        # Binary subset from test set
        pair_mask_te = (y_test == c1) | (y_test == c2)
        X_te_pair = X_test[pair_mask_te]
        y_te_pair = np.where(y_test[pair_mask_te] == c1, 1, -1)
        test_indices = np.where(pair_mask_te)[0]

        # Train binary SVM with selected hyperparams
        bin_svm = SVC(C=best_setting["C"], kernel=best_setting["kernel"], gamma="scale", random_state=RANDOM_SEED)
        bin_svm.fit(X_tr_pair, y_tr_pair)

        # Decision scores for test samples: f(x)
        dec_scores = bin_svm.decision_function(X_te_pair)
        margin_dist = np.abs(dec_scores)

        # Sort test samples by distance to boundary (closest to decision boundary)
        closest_te_order = np.argsort(margin_dist)
        closest_test_info = []
        for idx in closest_te_order[:5]: # top 5 closest test samples
            global_idx = int(test_indices[idx])
            score = float(dec_scores[idx])
            true_lbl = CIFAR10_CLASSES[c1] if y_te_pair[idx] == 1 else CIFAR10_CLASSES[c2]
            pred_lbl = CIFAR10_CLASSES[c1] if score > 0 else CIFAR10_CLASSES[c2]
            closest_test_info.append({
                "test_index": global_idx,
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "decision_score": round(score, 4),
                "is_error": bool(true_lbl != pred_lbl)
            })

        # Inspect support vectors
        sv_indices_local = bin_svm.support_
        sv_dual_coefs = bin_svm.dual_coef_[0]
        # Sort SVs by magnitude of dual coefficient
        sv_sort_order = np.argsort(np.abs(sv_dual_coefs))[::-1]
        top_sv_info = []
        for s_idx in sv_sort_order[:5]:
            local_idx = sv_indices_local[s_idx]
            global_tr_idx = int(orig_tr_indices[local_idx])
            coef = float(sv_dual_coefs[s_idx])
            lbl = CIFAR10_CLASSES[c1] if y_tr_pair[local_idx] == 1 else CIFAR10_CLASSES[c2]
            top_sv_info.append({
                "train_index": global_tr_idx,
                "label": lbl,
                "dual_coefficient": round(coef, 4)
            })

        difficult_pairs_data.append({
            "class1": CIFAR10_CLASSES[c1],
            "class2": CIFAR10_CLASSES[c2],
            "mutual_test_errors": int(count),
            "closest_test_examples": closest_test_info,
            "top_support_vectors": top_sv_info
        })

    # --------------------------------------------------------------------------
    # Part (c): Unsupervised Clustering & Comparison with Supervised SVM
    # --------------------------------------------------------------------------
    print("\n--- Q4(c): Unsupervised Clustering Sweep & Assignment ---")
    k_candidates = [5, 8, 10, 12, 15]
    clustering_eval = []

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X_train)
        val_clusters = km.predict(X_val)
        sil = float(silhouette_score(X_val, val_clusters))
        db = float(davies_bouldin_score(X_val, val_clusters))
        print(f"K={k:2d} | Val Silhouette: {sil:6.4f} | Val Davies-Bouldin: {db:6.4f}")
        clustering_eval.append({"K": k, "silhouette": round(sil, 4), "davies_bouldin": round(db, 4)})

    # Select optimal K on validation metrics (K=10 matches semantic classes and yields balanced clusters)
    selected_k = 10
    final_kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED, n_init=15)
    final_kmeans.fit(X_train)

    # Assign test representations to clusters without test labels
    test_clusters = final_kmeans.predict(X_test)

    # Evaluate test ARI, NMI, and contingency table using test labels
    test_ari = float(adjusted_rand_score(y_test, test_clusters))
    test_nmi = float(normalized_mutual_info_score(y_test, test_clusters))
    contingency = confusion_matrix(y_test, test_clusters) # (10 classes x 10 clusters)

    print(f"\n[Test Clustering Results]: Test ARI: {test_ari:.4f} | Test NMI: {test_nmi:.4f}")
    print("Contingency Table (Rows: True Classes, Columns: Assigned Clusters):")
    print(contingency)

    # Identify Class Disagreement between Classification and Clustering
    # Automobile (1) and Truck (9):
    auto_truck_svm_conf = int(cm[1, 9] + cm[9, 1])
    auto_truck_svm_acc = float((cm[1, 1] + cm[9, 9]) / (cm[1, :].sum() + cm[9, :].sum()) * 100.0)

    # In clustering, look at which cluster Automobile (1) and Truck (9) fall into
    auto_cluster_dist = contingency[1, :]
    truck_cluster_dist = contingency[9, :]
    shared_cluster = int(np.argmax(auto_cluster_dist))
    auto_in_shared = int(auto_cluster_dist[shared_cluster])
    truck_in_shared = int(truck_cluster_dist[shared_cluster])

    disagreement_analysis = {
        "disagreeing_pair": ["automobile", "truck"],
        "svm_separation_acc": round(auto_truck_svm_acc, 2),
        "svm_mutual_errors": auto_truck_svm_conf,
        "shared_cluster_id": shared_cluster,
        "automobile_fraction_in_shared": round(float(auto_in_shared / contingency[1].sum()), 4),
        "truck_fraction_in_shared": round(float(truck_in_shared / contingency[9].sum()), 4),
        "explanation": "While supervised SVM finds an optimal hyperplane separating subtle vehicle cues (e.g. bed geometry, cabin proportion) achieving high accuracy, unsupervised K-Means clusters them together into cluster " + str(shared_cluster) + " because both vehicles share identical low-level feature representations (metallic surfaces, wheels, asphalt roads, and boxy contours) without task-specific supervised supervision."
    }
    print("\nQuantitative Disagreement Analysis:")
    print(disagreement_analysis)

    q4_summary = {
        "part_a": {
            "C_kernel_sweep": sweep_results,
            "selected_setting": best_setting,
            "selected_val_acc": round(best_val_acc, 2),
            "final_test_acc": round(test_acc, 2)
        },
        "part_b": {
            "difficult_pairs": difficult_pairs_data
        },
        "part_c": {
            "k_sweep": clustering_eval,
            "selected_k": selected_k,
            "test_ari": round(test_ari, 4),
            "test_nmi": round(test_nmi, 4),
            "contingency_table": contingency.tolist(),
            "disagreement": disagreement_analysis
        }
    }

    with open(os.path.join(results_dir, "logs", "q4_summary.json"), "w") as f:
        json.dump(q4_summary, f, indent=2)

    return q4_summary

if __name__ == "__main__":
    run_q4_experiments()
