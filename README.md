# Physics-Guided Neural Network (PGNN) for VO2 Smart Window Inverse Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

## 📖 Overview
This repository contains the official source code and dataset for our study on the inverse design of VO2-based nanophotonic smart windows. 

We propose a **Physics-Guided Neural Network (PGNN)** to overcome the computational bottleneck of traditional 3D FDTD simulations. By embedding strict physical laws (thermodynamic directionality, energy conservation boundaries, and smoothness) directly into the loss function, our PGNN ensures physically realistic predictions while achieving an acceleration of over $10^5$ times compared to direct FDTD optimization.

## ✨ Key Features
To ensure strict scientific reproducibility and robustness, this codebase implements:
- **Strict Data Partitioning:** A rigorous `Split-then-Augment` protocol (Train/Validation/Test 3-way split) to eliminate data leakage and ensure an unbiased evaluation.
- **Physics-Informed Domain Randomization:** Gaussian perturbations are applied exclusively to the training set to account for fabrication-induced morphological irregularities (bridging the simulation-to-experiment gap).
- **Physics-Constrained Loss Topology:** Custom TensorFlow loss functions utilizing ReLU penalties to strictly enforce non-negativity ($\Delta \ge 0$) and boundary limits ($\Delta \le 1$).
- **Multi-Task Learning:** Simultaneously predicts both near-infrared solar modulation ($\Delta T_{NIR}$) and long-wave infrared emissivity variation ($\Delta \epsilon_{LWIR}$).

## 🗂️ Repository Structure
- `dataset.xlsx`: The raw ground-truth dataset generated via full-wave FDTD simulations (N=99).
- `main.py`: The unified Python script containing data preprocessing, PGNN model construction, the physics-constrained training pipeline, inverse design via Genetic Algorithm mapping, and visualization tools.
- `requirements.txt`: List of required Python packages for seamless environment setup.

## 🚀 Quick Start

### 1. Environment Setup
Clone this repository and install the required dependencies using `pip`:
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
pip install -r requirements.txt

```

### 2. Run the Pipeline

Execute the main script to reproduce the entire workflow, including model training, evaluation, and plotting:

```bash
python main.py

```

*(Note: Training the model typically takes less than a minute on a standard modern CPU).*

## 📊 Expected Outputs

Upon successful execution, the script will sequentially generate and display the following visualizations and metrics:

1. **Convergence Curves:** Dual-axis plots showing the standard MSE loss alongside the physical constraint penalties (Boundary & Smoothness), verifying the physics-guided learning process.
2. **Prediction Surfaces:** 3D surface mapping of the PGNN predictions evaluated against the ground-truth discrete FDTD data (Training and Testing sets).
3. **Inverse Design Analysis:** A comprehensive 4-in-1 chart including candidate design space distribution, comprehensive score contour maps, Pareto front analysis, and single-variable sensitivity curves.
4. **Accuracy Evaluation:** Scatter plots and $R^2$ scores evaluated strictly on the completely unseen testing partition.
5. **Feature Importance:** Permutation Feature Importance (PFI) analysis quantifying the relative impact of geometrical parameters within specified practical ranges.

## 📜 Data and Code Availability

The data and code presented in this repository are provided to ensure full transparency and reproducibility of the results discussed in the manuscript. The Testing partition is strictly sealed throughout the training phase and only utilized for the final $R^2$ accuracy evaluation.

```
```
