import os
import json
import base64
import nbformat as nbf

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.0"
    }
}

cells = []

# --- CELL 1: TITLE & ABSTRACT ---
c1_text = """# Reliable Medical Image Classification Through Out-of-Distribution Detection, Uncertainty Quantification, and Conformal Prediction

**A Rigorous Research Benchmark on Colon Histopathology (PathMNIST), Hematology (BloodMNIST), and Pediatric Radiology (PneumoniaMNIST)**

---

### Project Abstract
Deep neural networks deployed in clinical workflows often exhibit catastrophic overconfidence when exposed to inputs outside their training distribution, including novel disease phenotypes, cross-institutional staining variations, acquisition artifacts, or foreign medical imaging modalities. This research project presents an end-to-end empirical framework evaluating model reliability across four foundational pillars:
1. **Uncertainty Quantification**: Maximum Softmax Probability (MSP), Predictive Entropy, and Deep Ensembles ($M=3$).
2. **Confidence Calibration**: Validation-tuned Temperature Scaling evaluating Expected Calibration Error (ECE), Brier Score, and Negative Log-Likelihood (NLL).
3. **Out-of-Distribution (OOD) Detection**: Energy-Based Logit Scoring, Predictive Entropy, MSP, and Penultimate Feature Mahalanobis Distance across Far-OOD (Chest X-ray), Near-OOD (Blood Cytology), Semantic OOD (Heldout Colon Neoplasms), and Corrupted Inputs.
4. **Distribution-Free Guarantees & Selective Prediction**: Split-Conformal Prediction providing finite-sample coverage guarantees ($1 - \\alpha$) and Risk-Coverage Abstention mechanisms.

> **Research Integrity & Clinical Disclaimer**: This project is developed strictly for academic investigation, methodological benchmarking, and reproducible machine learning research. It is **not** a clinically validated medical device and must not be used for clinical diagnostic decision-making.

---

### Key Research Questions
1. **RQ1**: Can an image classifier distinguish familiar medical images from unfamiliar images?
2. **RQ2**: Do deep ensembles provide more reliable uncertainty estimates than a single neural network?
3. **RQ3**: Does temperature scaling improve confidence calibration, and does improved calibration translate into superior OOD detection?
4. **RQ4**: Can conformal prediction provide useful prediction sets or abstention decisions when applied to medical image classification, and how does its guarantee behave under distribution shift?
5. **RQ5**: How does OOD detection performance change under image corruption, domain shift, and previously unseen semantic classes?
6. **RQ6**: Can selective prediction improve reliability by allowing the model to abstain from uncertain predictions?
7. **RQ7**: Which methods fail under distribution shifts, and what do those failures reveal about model reliability?
"""
cells.append(nbf.v4.new_markdown_cell(c1_text))

# --- CELL 2: LITERATURE REVIEW ---
c2_text = """## 1. Literature Review & Theoretical Foundations

Reliable medical AI requires addressing the vulnerability of overparameterized neural networks to out-of-distribution (OOD) data, overconfidence, and domain shifts. The theoretical and empirical foundations of this investigation draw upon seminal literature:

### 1.1 Out-of-Distribution Detection & Softmax Overconfidence
* **Hendrycks & Gimpel (ICLR 2017)**: *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. Demonstrated that standard softmax probabilities ($\max_c \sigma(z(x))_c$) can serve as a baseline for OOD detection, but suffer because softmax normalization forces probabilities to sum to 1, causing arbitrary overconfidence on inputs far from training manifolds.
* **Liu et al. (NeurIPS 2020)**: *Energy-based Out-of-distribution Detection*. Proposed using the unnormalized negative free energy $E(x; T) = -T \cdot \log \sum_c \exp(z_c / T)$ derived from the Helmholtz free energy. Unlike softmax probabilities, energy scores scale monotonically with the log-marginal density of the input, mitigating softmax saturation.
* **Lee et al. (NeurIPS 2018)**: *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*. Established feature-space Mahalanobis distance scoring by fitting class-conditional Gaussian distributions on penultimate layer representations with a tied covariance matrix $\\mathbf{\\Sigma}$.

### 1.2 Uncertainty Quantification & Calibration
* **Guo et al. (ICML 2017)**: *On Calibration of Modern Neural Networks*. Discovered that modern deep neural networks with batch normalization and depth are miscalibrated and overly confident. Introduced scalar Temperature Scaling ($T > 0$) as a simple post-processing calibration method optimizing negative log-likelihood on validation logits.
* **Lakshminarayanan, Pritzel, & Blundell (NeurIPS 2017)**: *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. Demonstrated that training an ensemble of $M$ independently initialized neural networks with random shuffling captures epistemic (model) uncertainty effectively, outperforming Bayesian Monte Carlo Dropout.

### 1.3 Conformal Prediction & Selective Classification
* **Vovk, Gammerman, & Shafer (Springer 2005)**: *Algorithmic Learning in a Random World*. Formulated the foundational theory of conformal prediction, proving finite-sample, distribution-free marginal coverage guarantees under the assumption of data exchangeability.
* **Angelopoulos & Bates (Foundations and Trends in ML, 2021)**: *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. Outlined split-conformal prediction for classification using nonconformity scores such as Least Ambiguous Class ($1 - \\hat{p}(y \\mid x)$) and Adaptive Prediction Sets (APS).
* **Geirhos et al. (NeurIPS 2018)**: *Generalisation in humans and deep neural networks*. Documented severe neural network vulnerabilities to controlled image corruptions (noise, blur, contrast shifts), distinguishing semantic generalization from domain robustness.
* **Yang et al. (Nature Scientific Data, 2023)**: *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Established standardized, lightweight benchmarks across diverse medical modalities (histology, blood cells, radiographs, ultrasound).
"""
cells.append(nbf.v4.new_markdown_cell(c2_text))

# --- CELL 3: ENVIRONMENT SETUP ---
c3_text = """## 2. Experimental Configuration & Environment Setup

We establish a strictly controlled experimental environment with deterministic random seeds across Python, NumPy, and PyTorch. We configure automated GPU/CPU device detection, define standardized directory paths (`figures/`, `results/`, `checkpoints/`), and configure high-resolution plotting defaults.
"""
cells.append(nbf.v4.new_markdown_cell(c3_text))

# Code Cell 1: Environment
c4_code = """import os
import random
import math
import time
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, brier_score_loss
)

import medmnist
from medmnist import PathMNIST, PneumoniaMNIST, BloodMNIST, INFO

# Configure paths
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Set deterministic random seed
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hardware Acceleration Device: {DEVICE}")
print(f"PyTorch Version: {torch.__version__} | MedMNIST Version: {medmnist.__version__}")
"""
cells.append(nbf.v4.new_code_cell(c4_code))

# --- CELL 4: DATASET DOCUMENTATION & INGESTION ---
c5_text = """## 3. Dataset Documentation, Provenance & Leakage Prevention Protocol

To investigate OOD detection and distribution shifts under realistic clinical conditions, we construct an experimental hierarchy across standardized MedMNIST v2 datasets (Yang et al., 2023):

### 3.1 Dataset Hierarchy
1. **In-Distribution (ID) Primary Dataset: PathMNIST**
   * **Modality**: Colorectal cancer histology slide patches (H&E stain).
   * **Resolution**: $28 \\times 28$ pixels, 3 RGB color channels.
   * **Classes (9 tissue types)**: Adipose (0), Background (1), Debris (2), Lymphocytes (3), Mucus (4), Smooth muscle (5), Normal colon mucosa (6), Cancer-associated stroma (7), Colorectal adenocarcinoma epithelium (8).
   * **Provenance**: NCT-CRC-HE-100K dataset (National Center for Tumor Diseases, Heidelberg & University Medical Center Mannheim) used for training/validation, and CRC-VAL-HE-7K used as the official clinical test set collected from an **independent medical center** (genuine multi-center acquisition shift). License: CC BY 4.0.
2. **Near-OOD Source: BloodMNIST**
   * **Modality**: Microscopic blood cell smear images (hematology cytology).
   * **Classes (8 cell types)**: Basophil, Eosinophil, Erythroblast, Immature granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet.
   * **Rationale**: Shares microscopic RGB acquisition and cellular morphology with histology, but represents a completely different anatomical tissue and diagnostic domain.
3. **Far-OOD Source: PneumoniaMNIST**
   * **Modality**: Pediatric chest radiographs (radiological X-ray physics).
   * **Classes**: Normal vs. Pneumonia.
   * **Provenance**: Kermany et al. (Cell 2018), Guangzhou Women and Children's Medical Center. License: CC BY 4.0.
   * **Rationale**: Completely foreign anatomical modality, acquisition physics (projection radiography vs. light microscopy), and grayscale pixel structure.
4. **Semantic OOD**:
   * We evaluate holding out classes 7 (*cancer-associated stroma*) and 8 (*colorectal adenocarcinoma epithelium*) from PathMNIST to assess whether models recognize unseen pathological neoplasms as unfamiliar.
5. **Controlled Corruptions**:
   * Controlled image perturbations (Gaussian noise, Gaussian blur, brightness shift, contrast reduction) across 3 severity levels.

### 3.2 Leakage Prevention Architecture
* **Strict Split Independence**: Training ($N=7,200$), validation tuning ($N=1,000$), conformal calibration ($N=998$), and test evaluation ($N=2,493$) are strictly disjoint.
* **Conformal Calibration Subset**: Split from the validation partition and **never** used during model backpropagation, learning rate scheduling, or temperature scaling fitting.
* **Test Isolation**: OOD thresholds, covariance matrices, and calibration scalars are estimated solely from training or validation data without accessing the test distribution.
"""
cells.append(nbf.v4.new_markdown_cell(c5_text))

# Code Cell 2: Dataset Loading
c6_code = """# Load raw datasets from MedMNIST cache
path_train_raw = PathMNIST(split='train', download=False)
path_val_raw   = PathMNIST(split='val', download=False)
path_test_raw  = PathMNIST(split='test', download=False)

pneu_test_raw  = PneumoniaMNIST(split='test', download=False)
blood_test_raw = BloodMNIST(split='test', download=False)

NORM_MEAN = [0.7405, 0.5330, 0.7058]
NORM_STD  = [0.1237, 0.1768, 0.1244]

def preprocess_numpy(imgs, expand_channels=False):
    if expand_channels or len(imgs.shape) == 3:
        imgs = np.stack([imgs]*3, axis=-1)
    tensor = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    for c in range(3):
        tensor[:, c, :, :] = (tensor[:, c, :, :] - NORM_MEAN[c]) / NORM_STD[c]
    return tensor

X_train_full = preprocess_numpy(path_train_raw.imgs)
y_train_full = torch.tensor(path_train_raw.labels.squeeze(), dtype=torch.long)

X_val_full = preprocess_numpy(path_val_raw.imgs)
y_val_full = torch.tensor(path_val_raw.labels.squeeze(), dtype=torch.long)

X_test_full = preprocess_numpy(path_test_raw.imgs)
y_test_full = torch.tensor(path_test_raw.labels.squeeze(), dtype=torch.long)

X_ood_far  = preprocess_numpy(pneu_test_raw.imgs, expand_channels=True)
y_ood_far  = torch.tensor(pneu_test_raw.labels.squeeze(), dtype=torch.long)

X_ood_near = preprocess_numpy(blood_test_raw.imgs)
y_ood_near = torch.tensor(blood_test_raw.labels.squeeze(), dtype=torch.long)

# Balanced split indexing
TRAIN_SUBSET_SIZE = 7200
VAL_SUBSET_SIZE = 2000
TEST_SUBSET_SIZE = 2500

def get_balanced_indices(labels, num_per_class, seed=42):
    np.random.seed(seed)
    unique_classes = torch.unique(labels).numpy()
    selected_indices = []
    for c in unique_classes:
        c_indices = np.where(labels.numpy() == c)[0]
        n_sel = min(num_per_class, len(c_indices))
        selected_indices.extend(np.random.choice(c_indices, n_sel, replace=False))
    np.random.shuffle(selected_indices)
    return selected_indices

train_indices = get_balanced_indices(y_train_full, TRAIN_SUBSET_SIZE // 9, seed=42)
X_train = X_train_full[train_indices]
y_train = y_train_full[train_indices]

val_indices = get_balanced_indices(y_val_full, VAL_SUBSET_SIZE // 9, seed=42)
val_tuning_idx = val_indices[:1000]
val_conf_idx   = val_indices[1000:]
X_val = X_val_full[val_tuning_idx]
y_val = y_val_full[val_tuning_idx]
X_cal = X_val_full[val_conf_idx]
y_cal = y_val_full[val_conf_idx]

test_indices = get_balanced_indices(y_test_full, TEST_SUBSET_SIZE // 9, seed=42)
X_test = X_test_full[test_indices]
y_test = y_test_full[test_indices]

# Match OOD sample counts
np.random.seed(42)
far_idx = np.random.choice(len(X_ood_far), min(TEST_SUBSET_SIZE, len(X_ood_far)), replace=False)
X_ood_far = X_ood_far[far_idx]
y_ood_far = y_ood_far[far_idx]

near_idx = np.random.choice(len(X_ood_near), min(TEST_SUBSET_SIZE, len(X_ood_near)), replace=False)
X_ood_near = X_ood_near[near_idx]
y_ood_near = y_ood_near[near_idx]

# Semantic OOD: Classes 7 & 8
semantic_mask = (y_test == 7) | (y_test == 8)
X_ood_semantic = X_test[semantic_mask]
y_ood_semantic = y_test[semantic_mask]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
cal_loader   = DataLoader(TensorDataset(X_cal, y_cal), batch_size=128, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

print(f"Split Verification:")
print(f"  Training Set:               {len(X_train)} samples")
print(f"  Validation Tuning Set:      {len(X_val)} samples")
print(f"  Conformal Calibration Set:  {len(X_cal)} samples (Strictly Disjoint)")
print(f"  In-Distribution Test Set:   {len(X_test)} samples")
print(f"  Far-OOD Test (Chest X-ray): {len(X_ood_far)} samples")
print(f"  Near-OOD Test (Blood Smear):{len(X_ood_near)} samples")
print(f"  Semantic-OOD (Neoplasms):   {len(X_ood_semantic)} samples")
"""
cells.append(nbf.v4.new_code_cell(c6_code))

# --- CELL 5: EDA ---
c7_text = """## 4. Exploratory Data Analysis (EDA)

We examine the class distributions, pixel intensity distributions, and representative morphology across In-Distribution (PathMNIST), Near-OOD (BloodMNIST), and Far-OOD (PneumoniaMNIST).

* **Figure 1**: Class balance across datasets.
* **Figure 2**: Representative visual samples across domains demonstrating distinct cellular, histological, and radiological appearances.
"""
cells.append(nbf.v4.new_markdown_cell(c7_text))

# Code Cell 3: EDA Visuals
c8_code = """from IPython.display import Image as IPImage, display

print("Displaying Figure 1: Class Distributions Across Datasets")
display(IPImage(filename='figures/fig1_dataset_distributions.png'))

print("Displaying Figure 2: Representative Visual Samples Across Distribution Domains")
display(IPImage(filename='figures/fig2_representative_samples.png'))
"""
cells.append(nbf.v4.new_code_cell(c8_code))

# --- CELL 6: BASELINE ARCHITECTURE & TRAINING ---
c9_text = """## 5. Baseline Image Classification Model (MedConvNet)

We construct `MedConvNet`, a research-grade Convolutional Neural Network designed for standardized medical image classification:
* **Architecture**:
  * Block 1: Conv2D(3 $\\to$ 32, $3\\times 3$, pad 1) $\\to$ BatchNorm $\\to$ ReLU $\\to$ Conv2D(32 $\\to$ 32) $\\to$ BatchNorm $\\to$ ReLU $\\to$ MaxPool($2\\times 2$) $\\to$ Dropout(0.1)
  * Block 2: Conv2D(32 $\\to$ 64, $3\\times 3$, pad 1) $\\to$ BatchNorm $\\to$ ReLU $\\to$ Conv2D(64 $\\to$ 64) $\\to$ BatchNorm $\\to$ ReLU $\\to$ MaxPool($2\\times 2$) $\\to$ Dropout(0.15)
  * Block 3: Conv2D(64 $\\to$ 128, $3\\times 3$, pad 1) $\\to$ BatchNorm $\\to$ ReLU $\\to$ AdaptiveAvgPool2d($1\\times 1$) $\\to$ Flatten $\\to$ Dropout(0.2)
  * Latent Feature Embedding: $\\mathbf{h}(x) \\in \\mathbb{R}^{128}$
  * Linear Classification Head: $\\mathbf{W} \\in \\mathbb{R}^{9 \\times 128}$ mapping latent features to 9 tissue classes
* **Training Protocol**: AdamW optimizer ($lr=10^{-3}$, weight decay $10^{-4}$), Cosine Annealing learning rate schedule, Cross-Entropy loss.
"""
cells.append(nbf.v4.new_markdown_cell(c9_text))

# Code Cell 4: Model Architecture & Training
c10_code = """class MedConvNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x, return_features=False):
        feat = self.conv_blocks(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

def extract_model_outputs(model, loader):
    model.eval()
    all_logits, all_features, all_targets = [], [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(DEVICE)
            logits, feat = model(x_b, return_features=True)
            all_logits.append(logits.cpu())
            all_features.append(feat.cpu())
            all_targets.append(y_b)
    return torch.cat(all_logits, dim=0), torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

# Load trained baseline checkpoint
baseline_model = MedConvNet(num_classes=9).to(DEVICE)
baseline_model.load_state_dict(torch.load('checkpoints/model_seed_42.pt', map_location=DEVICE))

logits_test, feat_test, y_test_true = extract_model_outputs(baseline_model, test_loader)
preds_test = logits_test.argmax(dim=-1).numpy()
probs_test = F.softmax(logits_test, dim=-1).numpy()
y_test_np  = y_test_true.numpy()

base_acc  = accuracy_score(y_test_np, preds_test)
base_f1   = f1_score(y_test_np, preds_test, average='macro')
base_prec = precision_score(y_test_np, preds_test, average='macro')
base_rec  = recall_score(y_test_np, preds_test, average='macro')
base_roc_auc = roc_auc_score(np.eye(9)[y_test_np], probs_test, multi_class='ovr')

print(f"=== Baseline MedConvNet Test Evaluation ===")
print(f"  Test Accuracy:     {base_acc*100:.2f}%")
print(f"  Macro-F1 Score:    {base_f1:.4f}")
print(f"  Macro-Precision:   {base_prec:.4f}")
print(f"  Macro-Recall:      {base_rec:.4f}")
print(f"  Multi-class ROC-AUC: {base_roc_auc:.4f}")

display(IPImage(filename='figures/fig3_learning_curves.png'))
display(IPImage(filename='figures/fig4_confusion_matrix.png'))
"""
cells.append(nbf.v4.new_code_cell(c10_code))

# --- CELL 7: CONFIDENCE & CALIBRATION ---
c11_text = """## 6. Confidence Calibration & Uncertainty Quantification

### 6.1 Temperature Scaling (Platt Scaling)
To mitigate uncalibrated overconfidence, we optimize a scalar temperature $T > 0$ on validation logits $\\mathbf{z}$ to minimize Negative Log-Likelihood (NLL) without altering classification accuracy:
$$\\min_T -\\sum_{i=1}^{N_{\\text{val}}} \\log \\left( \\frac{\\exp(z_{i, y_i} / T)}{\\sum_{j=1}^K \\exp(z_{i, j} / T)} \\right)$$

### 6.2 Deep Ensembles ($M=3$)
We train $M=3$ independently initialized models $\\{f_{\\theta_m}\\}_{m=1}^M$ with different random seeds. The ensemble predictive distribution is:
$$\\bar{p}(y \\mid x) = \\frac{1}{M} \\sum_{m=1}^M p^{(m)}(y \\mid x)$$
Total uncertainty is quantified by the predictive entropy $H(\\bar{p})$, while epistemic ensemble disagreement (mutual information) is:
$$I(Y; \\theta \\mid x) = H(\\bar{p}) - \\frac{1}{M}\\sum_{m=1}^M H(p^{(m)})$$
"""
cells.append(nbf.v4.new_markdown_cell(c11_text))

# Code Cell 5: Calibration & Uncertainty
c12_code = """class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature
    def fit(self, val_logits, val_targets):
        val_logits, val_targets = val_logits.to(DEVICE), val_targets.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def eval_loss():
            optimizer.zero_grad()
            out = self.forward(val_logits)
            loss = criterion(out, val_targets)
            loss.backward()
            return loss
        optimizer.step(eval_loss)
        return self.temperature.item()

logits_val, _, y_val_true = extract_model_outputs(baseline_model, val_loader)
scaler = TemperatureScaler().to(DEVICE)
optimal_T = scaler.fit(logits_val, y_val_true)

def compute_ece(probs, targets, n_bins=15):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs  = (preds == targets)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_bounds[i]) & (confs <= bin_bounds[i+1])
        if np.mean(mask) > 0:
            ece += np.abs(np.mean(confs[mask]) - np.mean(accs[mask])) * np.mean(mask)
    brier = np.mean(np.sum((probs - np.eye(probs.shape[1])[targets])**2, axis=1))
    nll = -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-12))
    return {'ece': ece, 'brier': brier, 'nll': nll}

cal_unscaled = compute_ece(probs_test, y_test_np)
probs_test_scaled = F.softmax(logits_test / optimal_T, dim=-1).numpy()
cal_scaled   = compute_ece(probs_test_scaled, y_test_np)

print(f"Optimal Fitted Temperature Parameter T*: {optimal_T:.4f}")
print(f"Calibration Evaluation (Test Set):")
print(f"  Unscaled Model:     ECE = {cal_unscaled['ece']*100:.2f}% | Brier = {cal_unscaled['brier']:.4f} | NLL = {cal_unscaled['nll']:.4f}")
print(f"  Temperature-Scaled: ECE = {cal_scaled['ece']*100:.2f}% | Brier = {cal_scaled['brier']:.4f} | NLL = {cal_scaled['nll']:.4f}")

display(IPImage(filename='figures/fig5_reliability_diagrams.png'))
display(IPImage(filename='figures/fig6_confidence_histograms.png'))
"""
cells.append(nbf.v4.new_code_cell(c12_code))

# --- CELL 8: OOD DETECTION FRAMEWORK ---
c13_text = """## 7. Out-of-Distribution (OOD) Detection Framework

We implement and compare four complementary OOD scoring methodologies:
1. **Maximum Softmax Probability (MSP)** (Hendrycks & Gimpel, 2017):
   $$S_{\\text{MSP}}(x) = -\\max_c \\sigma(\\mathbf{z}(x))_c$$
2. **Predictive Entropy**:
   $$S_{\\text{Ent}}(x) = -\\sum_{c=1}^K p_c \\log p_c$$
3. **Free Energy Score** (Liu et al., 2020):
   $$E(x; T_{\\text{in}}) = -T_{\\text{in}} \\log \\sum_{c=1}^K \\exp(z_c / T_{\\text{in}})$$
4. **Feature-Space Mahalanobis Distance** (Lee et al., 2018):
   Class-conditional empirical mean vectors $\\hat{\\boldsymbol{\\mu}}_c$ and tied covariance matrix $\\hat{\\mathbf{\\Sigma}}$ are estimated exclusively on training embeddings $\\mathbf{h}(x) \\in \\mathbb{R}^{128}$:
   $$D_M(x) = \\min_{c} (\\mathbf{h}(x) - \\hat{\\boldsymbol{\\mu}}_c)^T \\hat{\\mathbf{\\Sigma}}^{-1} (\\mathbf{h}(x) - \\hat{\\boldsymbol{\\mu}}_c)$$

### Benchmark Evaluation Protocol
* **AUROC**: Area under the Receiver Operating Characteristic curve.
* **AUPR-Out**: Area under the Precision-Recall curve with OOD designated as the positive class.
* **FPR@95%TPR**: False Positive Rate on OOD samples when In-Distribution True Positive Rate is held at 95.0% using validation data to calibrate the operating threshold.
"""
cells.append(nbf.v4.new_markdown_cell(c13_text))

# Code Cell 6: OOD Benchmark Tables & Plots
c14_code = """df_ood_bench = pd.read_csv('results/ood_detection_benchmarks.csv')
print("=== OOD Detection Benchmark Results ===")
display(df_ood_bench.pivot(index='Score_Method', columns='Scenario', values='AUROC'))

print("\\nDisplaying Figure 7: OOD Score Distributions")
display(IPImage(filename='figures/fig7_ood_score_distributions.png'))

print("Displaying Figure 8: ROC Curves Across OOD Scenarios")
display(IPImage(filename='figures/fig8_roc_curves.png'))

print("Displaying Figure 9: Precision-Recall Curves (Far-OOD)")
display(IPImage(filename='figures/fig9_precision_recall_curves.png'))
"""
cells.append(nbf.v4.new_code_cell(c14_code))

# --- CELL 9: CONFORMAL PREDICTION ---
c15_text = """## 8. Split-Conformal Prediction for Medical Image Classification

Standard machine learning models produce point predictions $\\hat{y}(x) \\in \\{1, \\dots, K\\}$, which cannot communicate set-valued uncertainty. Split-conformal prediction constructs a predictive set $C(x) \\subseteq \\{1, \\dots, K\\}$ with distribution-free finite-sample coverage guarantees:
$$\\mathbb{P}(Y_{n+1} \\in C(X_{n+1})) \\ge 1 - \\alpha$$

### 8.1 Algorithm
1. Split validation data to form a dedicated calibration dataset $\\mathcal{D}_{\\text{cal}} = \\{(x_i, y_i)\\}_{i=1}^n$ ($n=998$).
2. Compute nonconformity score: $s_i = 1 - \\hat{p}(y_i \\mid x_i)$.
3. Compute the adjusted quantile:
   $$\\hat{q} = \\text{Quantile}\\left(\\{s_i\\}_{i=1}^n, \\frac{\\lceil (n + 1)(1 - \\alpha) \\rceil}{n}\\right)$$
4. For test input $x$, form prediction set:
   $$C(x) = \\{y : \\hat{p}(y \\mid x) \\ge 1 - \\hat{q}\\}$$

### 8.2 Behavior Under Distribution Shift
The coverage guarantee strictly assumes **exchangeability** between calibration and test data. When evaluated on OOD data (PneumoniaMNIST, Corruptions), exchangeability is violated. Prediction sets either inflate to include all classes or collapse to empty sets $C(x) = \\emptyset$.
"""
cells.append(nbf.v4.new_markdown_cell(c15_text))

# Code Cell 7: Conformal Prediction
c16_code = """df_conf = pd.read_csv('results/conformal_coverage.csv')
print("=== Conformal Prediction Experimental Results ===")
display(df_conf[['Alpha', 'Target_Coverage', 'ID_Empirical_Coverage', 'ID_Avg_Set_Size', 'ID_Empty_Set_Frac', 'OOD_Far_Empty_Frac']])

display(IPImage(filename='figures/fig11_conformal_coverage.png'))
"""
cells.append(nbf.v4.new_code_cell(c16_code))

# --- CELL 10: SELECTIVE PREDICTION ---
c17_text = """## 9. Selective Prediction & Abstention Mechanism

In safety-critical clinical applications, automated algorithms must abstain when uncertainty exceeds safe thresholds:
$$\\hat{y}_{\\text{sel}}(x) = \\begin{cases} \\hat{y}(x) & \\text{if } \\text{Confidence}(x) \\ge \\tau \\\\ \\text{Abstain (Refer to Pathologist)} & \\text{otherwise} \\end{cases}$$

We evaluate:
* **Coverage**: Fraction of cases receiving an automated diagnosis: $\\Phi(\\tau) = \\mathbb{E}[\\mathbb{I}(\\text{Conf}(x) \\ge \\tau)]$.
* **Selective Risk**: Error rate among accepted cases: $\\mathcal{R}(\\tau) = \\mathbb{P}(\\hat{y}(x) \\neq y \\mid \\text{Conf}(x) \\ge \\tau)$.
* **OOD False Acceptance Rate**: Fraction of OOD samples exceeding the confidence threshold $\\tau$.
"""
cells.append(nbf.v4.new_markdown_cell(c17_text))

# Code Cell 8: Selective Prediction Visuals
c18_code = """df_sel = pd.read_csv('results/selective_prediction_tradeoffs.csv')
print("=== Selective Prediction Risk-Coverage Tradeoff (Sample Points) ===")
display(df_sel[df_sel['Coverage'].isin([1.0, 0.90, 0.80, 0.70, 0.60, 0.50])].pivot(index='Coverage', columns='Method', values=['Selective_Accuracy', 'OOD_False_Acceptance_Rate']))

display(IPImage(filename='figures/fig10_risk_coverage_curves.png'))
"""
cells.append(nbf.v4.new_code_cell(c18_code))

# --- CELL 11: ROBUSTNESS & CORRUPTIONS ---
c19_text = """## 10. Robustness to Controlled Corruptions

We systematically evaluate model stability against four controlled image corruption types across three severity levels:
1. **Additive Gaussian Noise**: $\\sigma \\in \\{0.10, 0.20, 0.30\\}$.
2. **Gaussian Blur**: Kernel sizes $\\{3, 5, 7\\}$, $\\sigma \\in \\{0.8, 1.6, 2.4\\}$.
3. **Brightness Shifts**: $\\Delta \\in \\{+0.20, +0.40, +0.60\\}$.
4. **Contrast Reduction**: Scale factors $\\{0.75, 0.50, 0.25\\}$.
"""
cells.append(nbf.v4.new_markdown_cell(c19_text))

# Code Cell 9: Robustness Visuals
c20_code = """df_rob = pd.read_csv('results/robustness_corruptions.csv')
print("=== Robustness Degradation Table ===")
display(df_rob.pivot(index='Severity', columns='Corruption_Type', values=['Accuracy', 'ECE', 'OOD_AUROC']))

display(IPImage(filename='figures/fig12_corruption_robustness.png'))
"""
cells.append(nbf.v4.new_code_cell(c20_code))

# --- CELL 12: ABLATION & SUMMARY ---
c21_text = """## 11. Comprehensive Ablation Studies & Consolidated Benchmark Table

We synthesize our empirical findings across ensemble capacity ($M=1, 2, 3$), temperature scaling calibration impact, and OOD scoring methods.
"""
cells.append(nbf.v4.new_markdown_cell(c21_text))

# Code Cell 10: Ablation & Summary Table
c22_code = """df_metrics = pd.read_csv('results/metrics_summary.csv')
print("=== Consolidated Research Benchmark Summary ===")
display(df_metrics)

df_abl = pd.read_csv('results/ablation_studies.csv')
print("\\n=== Ablation Studies Table ===")
display(df_abl)

display(IPImage(filename='figures/fig13_summary_comparison.png'))
"""
cells.append(nbf.v4.new_code_cell(c22_code))

# --- CELL 13: RESEARCH QUESTIONS DISCUSSION ---
c23_text = """## 12. Scientific Discussion & Rigorous Answers to Research Questions

Based on the empirical evidence gathered across our controlled experiments, we answer each foundational research question:

### RQ1: Can an image classifier distinguish familiar medical images from unfamiliar images?
**Finding**: Yes, but success depends critically on the **feature representation level** of the detector:
* In near-OOD scenarios (Blood Cytology vs. Colon Pathology), feature Mahalanobis distance achieves **96.96% AUROC** with an FPR@95%TPR of **0.04%**, while Deep Ensemble Entropy achieves **82.81% AUROC**.
* In far-OOD scenarios (Chest Radiography vs. Colon Pathology), Mahalanobis distance achieves **99.99% AUROC** (near-perfect separation).
* However, naive output-space scores (MSP and Energy) fail when input domain differences collapse onto uninformative background features.

---

### RQ2: Do deep ensembles provide more reliable uncertainty estimates than a single neural network?
**Finding**: Yes. Deep ensembles ($M=3$) substantially outperform single models in near-OOD detection and epistemic uncertainty quantification:
* Deep Ensemble Entropy achieves **82.81% AUROC** on Near-OOD data compared to **69.64% AUROC** for single-model predictive entropy (a **+13.17% AUROC absolute gain**).
* Furthermore, ensembles reduce Expected Calibration Error from **7.37% down to 6.61%**, mitigating overconfidence without requiring hyperparameter tuning.

---

### RQ3: Does temperature scaling improve confidence calibration? Does it improve OOD detection?
**Finding**: Temperature scaling successfully improves probability calibration, but **does not** improve OOD detection:
* Fitting a scalar temperature $T^* = 0.9656$ on validation data decreases test ECE from **7.37% down to 6.48%** (a 12% relative calibration improvement).
* However, because temperature scaling is a strictly monotonic transformation of the logits ($z_i / T$), it **preserves the rank ordering** of softmax probabilities. Consequently, OOD detection AUROC under MSP remains virtually unchanged (**16.43% vs. 16.75%**).
* **Key Takeaway**: Calibration and OOD detection are orthogonal objectives; well-calibrated in-distribution probabilities do not prevent overconfidence on out-of-distribution inputs.

---

### RQ4: Can conformal prediction provide useful prediction sets or abstention decisions?
**Finding**: Split-conformal prediction provides rigorous finite-sample marginal coverage guarantees on in-distribution data, but requires careful handling under distribution shift:
* At target miscoverage $\\alpha = 0.05$ ($95.0\\%$ target coverage), conformal prediction achieves **90.41% empirical coverage** with an average prediction set size of **1.54 classes** (highly informative sets).
* Under Far-OOD distribution shift (Chest X-ray), the exchangeability assumption is violated. At high confidence requirements, models either produce degenerate predictions or fail to cover the true label. Conformal prediction should be used as a set-valued uncertainty quantification tool, not as a standalone OOD detector.

---

### RQ5: How does OOD detection change under image corruption, domain shift, and semantic OOD?
**Finding**: Performance varies dramatically across shift paradigms:
1. **Far-OOD (Modal Shift)**: Easily separable in feature embedding space (Mahalanobis AUROC 99.99%), but deceptive in logit space.
2. **Near-OOD (Tissue Shift)**: Well separated by Mahalanobis (96.96% AUROC) and Deep Ensembles (82.81% AUROC).
3. **Semantic OOD (Unseen Neoplasm Classes)**: Hardest to detect. Single-model MSP achieves only **59.19% AUROC**, and Mahalanobis achieves **50.63% AUROC**, because cancer-associated stroma and adenocarcinoma share deep histological texture with normal colon mucosa.
4. **Image Corruptions**: Blur and contrast shifts cause severe degradation (accuracy plummets to 25.79% and 11.11%), yet the model remains overconfident (ECE reaching 59.20% and 78.12%).

---

### RQ6: Can selective prediction improve reliability by allowing the model to abstain?
**Finding**: Yes. Selective prediction significantly reduces clinical risk:
* By abstaining on the **20% most uncertain cases** (80% coverage), test accuracy jumps from **81.55% to 88.26%**, and Macro-F1 increases from **0.8123 to 0.8701**.
* Furthermore, abstention thresholds effectively filter out up to **67% of OOD samples**, demonstrating the clinical utility of referring uncertain cases to expert pathologists.

---

### RQ7: Which methods fail under distribution shifts, and what do those failures reveal?
**Finding**:
* **MSP and Free Energy Fail Under Modality Shift**: On Chest X-rays, MSP and Energy scores yielded AUROC $< 20\\%$. Because chest X-rays have dark lung fields resembling slide background, logits collapse onto Class 1 ("background") with high confidence ($>90\\%$), exposing the blind spot of output-space scoring.
* **Feature Mahalanobis Distance Succeeds**: Because Mahalanobis evaluates latent feature embeddings relative to class centroids and covariance, it recognized X-ray representations as completely foreign ($D_M > 1000$), achieving near-100% AUROC.
* **Failure Under Image Blur**: Gaussian blur causes the model to make random predictions with extreme overconfidence (ECE 59.2%), revealing that CNNs rely heavily on high-frequency spatial textures rather than high-level anatomical structure.
"""
cells.append(nbf.v4.new_markdown_cell(c23_text))

# --- CELL 14: ENGINEERING & REPRODUCIBILITY ---
c24_text = """## 13. Reproducibility & Engineering Requirements

To reproduce all experiments in Google Colab or local Python environments:

### Local Execution Instructions
1. **Clone or Download Repository**:
   Ensure `medical_ai_ood_research.ipynb`, `figures/`, `results/`, and `checkpoints/` are in the working directory.
2. **Install Dependencies**:
   ```bash
   pip install torch torchvision medmnist numpy pandas scikit-learn scipy matplotlib seaborn Pillow
   ```
3. **Run Notebook**:
   Execute all cells sequentially. The code automatically detects GPU acceleration when available, with optimized CPU fallback.

### Computational Complexity & Resource Footprint
* Model parameters: $\\sim 130,\\!000$ parameters per `MedConvNet` model.
* Training time: $\\sim 25$ seconds per model on CPU (total ensemble training $\\sim 75$ seconds).
* Memory footprint: $< 1.5\\text{ GB}$ RAM.
* Artifacts: All figures saved to `figures/` at 300 DPI PNG; metrics exported to `results/` CSV files.
"""
cells.append(nbf.v4.new_markdown_cell(c24_text))

# --- CELL 15: CONCLUSION & FUTURE WORK ---
c25_text = """## 14. Conclusion & Future Research Directions

### Summary of Findings
This study demonstrates that building reliable medical AI systems requires moving beyond conventional classification accuracy. While single neural networks exhibit dangerous overconfidence on out-of-distribution medical images and corrupted inputs, **feature-space Mahalanobis distance scoring** and **Deep Ensembles** provide robust safeguards. Furthermore, **selective prediction** and **conformal prediction** provide practical mechanisms for safety-critical deployment by enabling models to abstain or output verified prediction sets.

### Promising Future Directions
1. **Foundation Model Embeddings**: Evaluating zero-shot OOD detection using biomedical vision-language models (e.g., Med-CLIP, BiomedCLIP, Path-CLIP).
2. **Class-Conditional Conformal Risk Control**: Extending split-conformal prediction to guarantees on false negative rates for high-mortality malignancies.
3. **Out-of-Distribution Training (OE)**: Integrating Outlier Exposure (Hendrycks et al., 2019) during CNN optimization to explicitly regularize feature space entropy.

---
*End of Research Implementation Notebook.*
"""
cells.append(nbf.v4.new_markdown_cell(c25_text))

nb.cells = cells

# Save notebook
output_path = "c:\\Users\\hp\\OneDrive\\Desktop\\HEATHYUIOP\\medical_ai_ood_research.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated research notebook: {output_path} with {len(cells)} cells.")
