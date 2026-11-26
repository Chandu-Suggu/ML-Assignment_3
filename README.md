# Machine Learning Assignment 3: MLPs, CNNs, Autoencoders, SVMs, and Clustering on CIFAR-10

**Course**: Machine Learning  
**Student Name**: Suggu Chandu  
**Roll Number**: M26CSA017  
**Department**: Computer Science and Engineering, Indian Institute of Technology Jodhpur  
**Email**: m26csa017@iitj.ac.in  
**Repository**: [https://github.com/Chandu-Suggu/ML-Assignment_3](https://github.com/Chandu-Suggu/ML-Assignment_3)  
**Submission Document**: `report/report.pdf` (CVPR 2-Column Format, 8 Pages)  

---

## 1. Executive Summary of Experimental Results

All experiments strictly follow the assignment guidelines:
- **Stratified Split**: 45,000 train (4,500/class) and 5,000 validation (500/class) split generated using random seed **26017** (numeric characters of roll number `M26CSA017` in order).
- **Official Test Set**: 10,000 official CIFAR-10 test images, evaluated strictly once per final selected model without any test-time tuning.
- **Fair Comparisons**: All model comparisons are strictly matched by parameter budgets and identical training pipelines.

### Key Deliverables Across Questions:

| Question | Topic | Model / Setting | Trainable Params | Val Metric | Official Test Metric | Key Finding / Conclusion |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Q1(a)** | MLP Failure Mode | Deep Global MLP (4-layer) | 521,898 | 55.10% Acc | 53.99% Acc | Shallow vs Deep difference is only +0.08% |
| | | Shallow Wide MLP (2-layer) | 518,290 | 54.58% Acc | 53.91% Acc | Depth alone does not resolve MLP visual deficit |
| | | Locally-Connected Patch MLP | **510,790** | **62.80% Acc** | **61.92% Acc** | **+7.93% gain** proving lack of 2D spatial locality is root cause |
| **Q1(b)** | Memorisation | Deep MLP on 40% Label Noise | 521,898 | 47.38% (Peak Ep 4) | 41.11% Acc | **Simplicity Bias**: Fits real structure first before memorising |
| **Q2(a)** | CNN vs MLP | ConvClassifier vs Deep MLP | 571k vs 522k | 85.42% vs 55.10% | **85.00% vs 53.99%** | **+31.01% gain** for CNN under fair parameter budget |
| **Q2(b)** | Spatial Perturbation | Fixed Spatial Coordinate Permutation | 571k vs 522k | 51.00% vs 55.00% | **51.13% vs 53.85%** | **CNN collapses (-33.87%)** while MLP is invariant (-0.14%) |
| **Q2(c)** | Occlusion & Robustness | Center Occlusion (12x12 cutout) | 571k vs 522k | 79.46% vs 52.42% | **78.12% vs 51.71%** | CNN retains high distributed contextual robustness |
| | | Spatial Translation (+4px shift) | 571k vs 522k | --- | **65.80% vs 29.77%** | MLP suffers catastrophic failure (-24.22%) |
| **Q3(a)** | Autoencoders | AE Bottleneck 64 | 523,619 | 0.133893 MSE | 0.132700 MSE | Compression factor 48x |
| | | AE Bottleneck 256 | 1,310,243 | 0.072457 MSE | 0.071954 MSE | Compression factor 12x |
| | | **AE Bottleneck 1024** | **4,456,739** | **0.057038 MSE** | **0.056631 MSE** | **Selected on Val MSE** (57.4% error reduction) |
| **Q3(b)** | Denoising Autoencoder | Gaussian Noise ($\sigma=0.25$) + DAE-256 | 1,310,243 | 0.076158 MSE | 0.075579 MSE | Reconstructions drop CNN accuracy (50.21% $\to$ 29.87%) |
| | | Decision on Downstream Info | --- | --- | --- | **MSE smoothing acts as low-pass filter, destroying edge cues** |
| **Q3(c)** | Class Info vs Rec | Linear Softmax Probes on Frozen Reps | 64 / 256 / 1024 | 36.5% / 44.4% / 51.6% | **36.51% / 44.44% / 51.59%** | Monotonic gain with bottleneck capacity |
| | | DAE-256 vs Clean AE-256 | 256 | 45.3% vs 44.4% | **45.28% vs 44.44%** | **DAE has higher class acc despite worse MSE (divergence)** |
| **Q4(a)** | SVM Sweep | Fixed 128-d CNN Representation | $N_{\text{tr}}=6\text{k}, N_{\text{val}}=1\text{k}$ | **84.90% Val Acc** | **84.96% Test Acc** | **Selected**: Linear Kernel, $C=0.01$ (SV fraction: 15.4%) |
| **Q4(b)** | Boundary Analysis | Cat vs Dog / Bird vs Cat / Bird vs Deer | Binary SVMs | Margin $|f(x)| \approx 0$ | 233 / 98 / 97 errors | Quadruped silhouette overlap & ambiguous head contours |
| **Q4(c)** | Unsupervised Clustering | K-Means ($K=10$ selected on Val) | $K=10$ | Sil: 0.2245, DB: 1.519 | **ARI: 0.6974, NMI: 0.7087** | Supervised SVM separates Auto vs Truck (**91.15%**), |
| | | Quantitative Disagreement | --- | --- | --- | **K-Means clusters 91.3% together** due to shared road/metal features |

---

## 2. Directory Structure

```
ML-3/
├── data/
│   ├── split_indices.npz         # Saved stratified 45k train / 5k val indices (seed 26017)
│   └── cifar-10-batches-py/      # Extracted CIFAR-10 python dataset
├── src/
│   ├── dataset.py                # Ultra-fast in-memory Tensor dataloaders & split verification
│   ├── models.py                 # PyTorch architectures (DeepMLP, WideMLP, LCN, CNN, Autoencoders)
│   ├── train_utils.py            # Modular training, loss tracking, evaluation routines
│   ├── q1_experiments.py         # Q1(a) Architecture comparisons & Q1(b) Memorisation dynamics
│   ├── q2_experiments.py         # Q2(a) Clean benchmark, Q2(b) Permutation, Q2(c) Occlusion/Translation
│   ├── q3_experiments.py         # Q3(a) 3 Autoencoders, Q3(b) Denoising, Q3(c) Linear probing
│   ├── q4_experiments.py         # Q4(a) SVM C/Kernel sweep, Q4(b) Boundaries, Q4(c) Clustering
│   └── generate_report_assets.py # Publication-quality vector figure generation
├── results/
│   ├── checkpoints/              # Saved model checkpoints and training history JSONs
│   ├── logs/                     # Complete experimental summary JSONs (Q1, Q2, Q3, Q4)
│   └── figures/                  # 9 Publication figures (both vector PDF and high-res PNG)
├── report/
│   ├── main.typ                  # Typst source document for CVPR 2-column report
│   ├── references.bib            # BibTeX bibliography with authoritative references
│   └── report.pdf                # Compiled 8-page CVPR format submission PDF
├── requirements.txt              # Pip dependencies
├── environment.yml               # Conda environment definition
├── run_all.sh                    # End-to-end execution script (Bash)
├── run_all.ps1                   # End-to-end execution script (PowerShell)
└── README.md                     # Complete project documentation (this file)
```

---

## 3. Reproduction Instructions

### 3.1 Environment Setup
```bash
# Option A: Conda
conda env create -f environment.yml
conda activate ml3_cifar10

# Option B: Pip
pip install -r requirements.txt
```

### 3.2 Single-Command Full Reproduction
To run all experiments, generate all figures, and compile the final report PDF:
```bash
# On Linux / macOS:
bash run_all.sh

# On Windows (PowerShell):
powershell -ExecutionPolicy Bypass -File ./run_all.ps1
```

### 3.3 Individual Module Execution
```bash
# 1. Dataset split verification
python -c "from src.dataset import get_or_create_stratified_split; get_or_create_stratified_split()"

# 2. Q1 Experiments (MLP comparisons & memorisation)
python -u -m src.q1_experiments

# 3. Q2 Experiments (CNN vs MLP, spatial permutation, occlusion)
python -u -m src.q2_experiments

# 4. Q3 Experiments (Convolutional autoencoders, denoising, linear probes)
python -u -m src.q3_experiments

# 5. Q4 Experiments (SVM sweep, support vectors, unsupervised clustering)
python -u -m src.q4_experiments

# 6. Generate all figures
python -u -m src.generate_report_assets

# 7. Compile CVPR Report PDF
python -c "import typst; typst.compile('report/main.typ', output='report/report.pdf', root='.')"
```
