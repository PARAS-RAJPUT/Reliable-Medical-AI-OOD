# Reliable Medical Image Classification Through Out-of-Distribution Detection, Uncertainty Quantification, and Conformal Prediction

**A Research Benchmark on Colon Histopathology (PathMNIST), Hematology (BloodMNIST), and Pediatric Radiology (PneumoniaMNIST)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![MedMNIST v3.0+](https://img.shields.io/badge/MedMNIST-3.0.2-green.svg)](https://medmnist.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

> **Academic Research & Non-Diagnostic Disclaimer**:
> This project is developed exclusively for academic research, reproducible experimentation, and scientific investigation into deep learning reliability under distribution shifts. **It is not a clinically validated diagnostic device and must not be used for primary patient diagnosis or clinical treatment planning.**

---

## 1. Project Overview

Deep neural networks deployed in clinical decision-support systems frequently encounter inputs divergent from their training distributions—including novel tissue phenotypes, cross-institutional staining variations, acquisition artifacts, and foreign imaging modalities. Standard classifiers trained with cross-entropy loss often yield catastrophically high softmax confidence on unrecognizable or corrupt inputs.

This project implements an end-to-end research benchmark investigating model reliability, confidence calibration, out-of-distribution (OOD) detection, conformal prediction, and selective classification across standardized medical imaging datasets:
* **Primary In-Distribution (ID)**: **PathMNIST** (Colorectal adenocarcinoma histology patches, 9 tissue classes, NCT-CRC-HE-100K train/val and multi-center CRC-VAL-HE-7K test set).
* **Near-OOD Source**: **BloodMNIST** (Microscopic blood cytology, 8 cell classes).
* **Far-OOD Source**: **PneumoniaMNIST** (Pediatric chest radiographs, CC BY 4.0, Kermany et al., *Cell* 2018).
* **Semantic-OOD Source**: PathMNIST held-out neoplastic classes (cancer-associated stroma and adenocarcinoma epithelium).
* **Controlled Corruptions**: Additive Gaussian noise, Gaussian blur, brightness shifts, and contrast attenuations across 3 severity levels.

### Visualizing Distribution Domains
The model is tested on colorectal histology (in-distribution), microscopic blood cell cytology (near-OOD), and pediatric chest radiographs (far-OOD):

<p align="center">
  <img src="figures/fig2_representative_samples.png" alt="Representative Samples Across Distribution Domains" width="95%"/>
</p>

---

## 2. Key Research Findings Summary

| Method / Strategy | Accuracy (%) | Macro-F1 | ECE (%) | Far-OOD AUROC (%) | Near-OOD AUROC (%) | Primary Role in Reliable Medical AI |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Single CNN (MSP)** | 81.55% | 0.8123 | 7.37% | 16.43% | 66.36% | Standard classification baseline; prone to background collapse |
| **Temperature-Scaled CNN ($T=0.97$)** | 81.55% | 0.8123 | **6.48%** | 16.75% | 66.36% | Optimizes probability calibration; does not improve OOD rank |
| **Free Energy Scoring** | 81.55% | 0.8123 | 7.37% | 0.06% | 74.68% | Output-space logit scoring; outperforms MSP on Near-OOD |
| **Feature Mahalanobis Distance** | 81.55% | 0.8123 | 7.37% | **99.99%** | **96.96%** | **Top OOD Detector**; detects foreign features in latent space |
| **Deep Ensemble ($M=3$)** | 80.83% | 0.8023 | 6.61% | 37.00% | **82.81%** | Epistemic uncertainty & disagreement estimation |
| **Selective Prediction (80% Cov)** | **88.26%** | **0.8701** | — | — | — | Abstains on uncertain cases; error drops from 18.5% to 11.7% |
| **Split Conformal Prediction ($\alpha=0.05$)** | **90.41%** (Cov) | — | — | — | — | Prediction sets (avg size 1.54) with finite-sample guarantees |

### Method Comparison Across OOD Regimes

<p align="center">
  <img src="figures/fig13_summary_comparison.png" alt="OOD Detection Performance Comparison Across Method Families" width="85%"/>
</p>

---

## 3. Visual Gallery of Research Experiments

### 3.1 Exploratory Data Analysis & Class Distributions
<p align="center">
  <img src="figures/fig1_dataset_distributions.png" alt="Dataset Class Distributions" width="95%"/>
</p>

### 3.2 Baseline Performance & Confusion Matrix
<p align="center">
  <img src="figures/fig3_learning_curves.png" alt="Learning Curves" width="48%"/>
  <img src="figures/fig4_confusion_matrix.png" alt="Confusion Matrix" width="48%"/>
</p>

### 3.3 Confidence Calibration: Reliability Diagrams & Confidence Histograms
Temperature scaling ($T^* = 0.9656$) drops ECE from **7.37% to 6.48%**:
<p align="center">
  <img src="figures/fig5_reliability_diagrams.png" alt="Reliability Diagrams" width="90%"/>
</p>
<p align="center">
  <img src="figures/fig6_confidence_histograms.png" alt="Confidence Histograms Across Domains" width="75%"/>
</p>

### 3.4 Out-of-Distribution Detection: Score Distributions & ROC Curves
Penultimate latent feature Mahalanobis distance cleanly separates both near-OOD and far-OOD medical images:
<p align="center">
  <img src="figures/fig7_ood_score_distributions.png" alt="OOD Score Distributions" width="90%"/>
</p>
<p align="center">
  <img src="figures/fig8_roc_curves.png" alt="ROC Curves Across OOD Scenarios" width="90%"/>
</p>
<p align="center">
  <img src="figures/fig9_precision_recall_curves.png" alt="Precision-Recall Curves" width="70%"/>
</p>

### 3.5 Selective Prediction: Risk-Coverage Abstention Tradeoffs
Abstaining on the 20% most uncertain predictions increases accuracy from **81.55% to 88.26%**:
<p align="center">
  <img src="figures/fig10_risk_coverage_curves.png" alt="Risk-Coverage Curves" width="90%"/>
</p>

### 3.6 Split-Conformal Prediction & Coverage Under Shift
Empirical coverage vs. target coverage ($1 - \alpha$) and set efficiency:
<p align="center">
  <img src="figures/fig11_conformal_coverage.png" alt="Conformal Coverage Analysis" width="90%"/>
</p>

### 3.7 Robustness Across Controlled Corruptions
Diagnostic accuracy collapses under blur (25.79%) and contrast shifts (11.11%), while overconfidence surges (ECE up to 78.12%):
<p align="center">
  <img src="figures/fig12_corruption_robustness.png" alt="Robustness Under Image Corruptions" width="90%"/>
</p>

---

## 4. Answers to Core Research Questions

1. **Distinguishing Familiar from Unfamiliar Images (RQ1)**: Yes. Feature-space Mahalanobis distance separates Far-OOD radiographs with **99.99% AUROC** and Near-OOD blood cells with **96.96% AUROC** ($0.04\%$ FPR95). However, naive output-space softmax probabilities fail when foreign textures trigger uninformative background logits.
2. **Deep Ensembles vs. Single Networks (RQ2)**: Deep Ensembles ($M=3$) achieve **82.81% AUROC** on Near-OOD cytology compared to **69.64%** for single-model predictive entropy (+13.17% gain), while reducing Expected Calibration Error from 7.37% to 6.61%.
3. **Temperature Scaling Impact (RQ3)**: Validation-tuned temperature scaling ($T=0.9656$) reduces ECE from 7.37% to 6.48% (a 12% relative calibration improvement). However, because temperature scaling is a strictly monotonic logit transformation, it preserves score rankings and yields essentially identical OOD AUROC (16.43% vs. 16.75%). Confidence calibration and OOD detection are orthogonal objectives.
4. **Conformal Prediction Guarantees (RQ4)**: Split-conformal prediction provides finite-sample marginal coverage on in-distribution data (90.41% empirical coverage at $\alpha=0.05$ with an average set size of 1.54 classes). Under Far-OOD distribution shift, exchangeability breaks, demonstrating that conformal prediction must be treated as a set-valued uncertainty quantification method rather than a standalone OOD classifier.
5. **Distribution Shift Taxonomy (RQ5)**: Models behave fundamentally differently under modality shift (Far-OOD, easily separated in latent space), anatomical shift (Near-OOD, separated by ensembles and Mahalanobis), semantic holdout (subtle histological overlap, hard to separate), and corruptions (severe accuracy collapse with surging overconfidence).
6. **Selective Prediction & Abstention (RQ6)**: Allowing the classifier to abstain on the 20% most uncertain predictions increases diagnostic accuracy from **81.55% to 88.26%** and Macro-F1 from **0.8123 to 0.8701**, while filtering out over 65% of OOD samples.
7. **Failure Modes & Reliability Implications (RQ7)**: Output-space methods (MSP and Energy) fail catastrophically on Chest X-rays because dark radiographs collapse onto Class 1 ("slide background") with 90%+ confidence. Latent feature space evaluation (Mahalanobis distance) is essential for robust deployment. Furthermore, Gaussian blur degrades accuracy to 25.79% while ECE balloons to 59.20%, proving that CNNs over-rely on high-frequency spatial textures.

---

## 5. Repository Structure

```
HEATHYUIOP/
├── medical_ai_ood_research.ipynb     # Complete, executed research Jupyter notebook
├── run_experiments.py               # Standalone Python experimental pipeline runner
├── build_notebook.py                # Notebook synthesis and formatting utility
├── README.md                        # Documentation and research report
├── figures/                         # 13 high-resolution publication figures (300 DPI)
│   ├── fig1_dataset_distributions.png
│   ├── fig2_representative_samples.png
│   ├── fig3_learning_curves.png
│   ├── fig4_confusion_matrix.png
│   ├── fig5_reliability_diagrams.png
│   ├── fig6_confidence_histograms.png
│   ├── fig7_ood_score_distributions.png
│   ├── fig8_roc_curves.png
│   ├── fig9_precision_recall_curves.png
│   ├── fig10_risk_coverage_curves.png
│   ├── fig11_conformal_coverage.png
│   ├── fig12_corruption_robustness.png
│   └── fig13_summary_comparison.png
├── results/                         # Exported benchmark CSV metric tables
│   ├── metrics_summary.csv
│   ├── ood_detection_benchmarks.csv
│   ├── conformal_coverage.csv
│   ├── selective_prediction_tradeoffs.csv
│   ├── robustness_corruptions.csv
│   └── ablation_studies.csv
└── checkpoints/                     # Serialized PyTorch model weights
    ├── model_seed_42.pt
    ├── model_seed_142.pt
    └── model_seed_242.pt
```

---

## 6. Quickstart & Reproducibility Instructions

### 6.1 Environment Installation
```bash
pip install torch torchvision medmnist numpy pandas scikit-learn scipy matplotlib seaborn Pillow jupyter
```

### 6.2 Running the Jupyter Notebook
Launch JupyterLab or Jupyter Notebook:
```bash
jupyter lab medical_ai_ood_research.ipynb
```
The notebook is completely pre-executed and self-contained with all cells, plots, tables, and narrative interpretations.

### 6.3 Executing via Standalone CLI Script
To re-run the full training, calibration, and benchmarking pipeline from scratch:
```bash
python run_experiments.py
```

---

## 7. Scholarly References

1. **Hendrycks, D., & Gimpel, K.** (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. International Conference on Learning Representations (ICLR).
2. **Liu, W., Wang, X., Owens, J. D., & Li, Y.** (2020). *Energy-based Out-of-distribution Detection*. Advances in Neural Information Processing Systems (NeurIPS), 33.
3. **Lee, K., Lee, K., Lee, H., & Shin, J.** (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*. Advances in Neural Information Processing Systems (NeurIPS), 31.
4. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks*. International Conference on Machine Learning (ICML), PMLR 70.
5. **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. Advances in Neural Information Processing Systems (NeurIPS), 30.
6. **Angelopoulos, A. N., & Bates, S.** (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
7. **Vovk, V., Gammerman, A., & Shafer, G.** (2005). *Algorithmic Learning in a Random World*. Springer Science & Business Media.
8. **Yang, J., Shi, R., Wei, D., et al.** (2023). *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Nature Scientific Data, 10(1), 41.
9. **Kermany, D. S., Goldbaum, M., Cai, W., et al.** (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122-1131.
10. **Geirhos, R., Temme, C. R., Rauber, J., et al.** (2018). *Generalisation in humans and deep neural networks*. Advances in Neural Information Processing Systems (NeurIPS), 31.
