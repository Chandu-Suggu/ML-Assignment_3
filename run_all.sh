#!/bin/bash
# ==============================================================================
# Complete Reproduction Script for Machine Learning Assignment 3
# Student: Suggu Chandu | Roll Number: M26CSA017 | Seed: 26017
# ==============================================================================
set -e

echo "=== Machine Learning Assignment 3: Full Reproduction Pipeline ==="
echo "Student: Suggu Chandu (Roll No: M26CSA017)"
echo "Stratified Train/Val Split Random Seed: 26017"

# Step 1: Create Stratified Split (45k train, 5k val) using Roll Number Seed
echo "\n[Step 1] Verifying Stratified Split..."
python -c "from src.dataset import get_or_create_stratified_split; get_or_create_stratified_split()"

# Step 2: Execute Q1 Experiments (MLP Comparisons & Memorisation)
echo "\n[Step 2] Executing Q1 Experiments..."
python -u -m src.q1_experiments

# Step 3: Execute Q2 Experiments (CNN vs MLP, Spatial Permutation, Occlusion)
echo "\n[Step 3] Executing Q2 Experiments..."
python -u -m src.q2_experiments

# Step 4: Execute Q3 Experiments (Convolutional Autoencoders, DAE, Linear Probing)
echo "\n[Step 4] Executing Q3 Experiments..."
python -u -m src.q3_experiments

# Step 5: Execute Q4 Experiments (SVM C/Kernel Sweep, Boundary Support Vectors, Clustering)
echo "\n[Step 5] Executing Q4 Bonus Experiments..."
python -u -m src.q4_experiments

# Step 6: Render all Publication-Quality Figures & Visual Artifacts
echo "\n[Step 6] Rendering Publication Figures..."
python -u -m src.generate_report_assets

# Step 7: Compile CVPR Report PDF
echo "\n[Step 7] Compiling CVPR Two-Column PDF Report..."
python -c "import typst; typst.compile('report/main.typ', output='report/report.pdf')"

echo "\n=== All Experiments and Deliverables Successfully Completed! ==="
