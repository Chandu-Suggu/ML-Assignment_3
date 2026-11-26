# ==============================================================================
# Complete Reproduction Script for Machine Learning Assignment 3 (PowerShell)
# Student: Suggu Chandu | Roll Number: M26CSA017 | Seed: 26017
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "=== Machine Learning Assignment 3: Full Reproduction Pipeline ===" -ForegroundColor Cyan
Write-Host "Student: Suggu Chandu (Roll No: M26CSA017)" -ForegroundColor Yellow
Write-Host "Stratified Train/Val Split Random Seed: 26017" -ForegroundColor Yellow

# Step 1: Create Stratified Split (45k train, 5k val) using Roll Number Seed
Write-Host "`n[Step 1] Verifying Stratified Split..." -ForegroundColor Green
python -c "from src.dataset import get_or_create_stratified_split; get_or_create_stratified_split()"

# Step 2: Execute Q1 Experiments (MLP Comparisons & Memorisation)
Write-Host "`n[Step 2] Executing Q1 Experiments..." -ForegroundColor Green
python -u -m src.q1_experiments

# Step 3: Execute Q2 Experiments (CNN vs MLP, Spatial Permutation, Occlusion)
Write-Host "`n[Step 3] Executing Q2 Experiments..." -ForegroundColor Green
python -u -m src.q2_experiments

# Step 4: Execute Q3 Experiments (Convolutional Autoencoders, DAE, Linear Probing)
Write-Host "`n[Step 4] Executing Q3 Experiments..." -ForegroundColor Green
python -u -m src.q3_experiments

# Step 5: Execute Q4 Experiments (SVM C/Kernel Sweep, Boundary Support Vectors, Clustering)
Write-Host "`n[Step 5] Executing Q4 Bonus Experiments..." -ForegroundColor Green
python -u -m src.q4_experiments

# Step 6: Render all Publication-Quality Figures & Visual Artifacts
Write-Host "`n[Step 6] Rendering Publication Figures..." -ForegroundColor Green
python -u -m src.generate_report_assets

# Step 7: Compile CVPR Report PDF
Write-Host "`n[Step 7] Compiling CVPR Two-Column PDF Report..." -ForegroundColor Green
python -c "import typst; typst.compile('report/main.typ', output='report/report.pdf')"

Write-Host "`n=== All Experiments and Deliverables Successfully Completed! ===" -ForegroundColor Cyan
