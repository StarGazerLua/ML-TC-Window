# ML-TC-Window
# Physics-Guided Neural Network (PGNN) for VO2 Smart Window Inverse Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

## 📖 Overview
This repository contains the official source code and dataset for our study on the inverse design of VO2-based nanophotonic smart windows. 

We propose a **Physics-Guided Neural Network (PGNN)** to overcome the computational bottleneck of traditional 3D FDTD simulations. By embedding strict physical laws (thermodynamic directionality, energy conservation boundaries, and smoothness) directly into the loss function, our PGNN ensures physically realistic predictions while achieving an acceleration of over $10^5$ times compared to direct FDTD optimization.

## ✨ Key Features
To ensure strict scientific reproducibility and robustness, this codebase implements:
- **Strict Data Partitioning:** A rigorous `Split-then-Augment` protocol (Train/Validation/Test 3-way split) to eliminate data leakage.
- **Physics-Informed Domain Randomization:** Gaussian perturbations applied exclusively to the training set to account for fabrication-induced morphological irregularities.
- **Physics-Constrained Loss Topology:** Custom TensorFlow loss functions utilizing ReLU penalties to strictly enforce non-negativity ($\Delta \ge 0$) and boundary limits ($\Delta \le 1$).

## 🗂️ Repository Structure
- `dataset.xlsx`: The raw ground-truth dataset generated via full-wave FDTD simulations.
- `main.py`: The unified Python script containing data preprocessing, PGNN model construction, training pipeline, inverse design via Genetic Algorithm, and visualization.
- `requirements.txt`: List of required Python packages.

## 🚀 Quick Start

### 1. Environment Setup
Clone this repository and install the required dependencies:
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
pip install -r requirements.txt
