# Efficient GAN: Stability & Data Efficiency on Limited CIFAR-10

This repository contains the implementation of a modernized **DCGAN** designed to address two major challenges in Generative Adversarial Networks: **training instability** and **data scarcity**.

I demonstrate that by combining modern regularization techniques (**R1 Gradient Penalty**) with data efficiency methods (**Differentiable Augmentation**), I can train high-fidelity generators on just **20% of the CIFAR-10 dataset**. Furthermore, I introduce a novel **Adaptive Label Smoothing (ALS)** mechanism that dynamically stabilizes training, outperforming standard static regularization methods.

## 📊 Key Results

I evaluated my models using the **Fréchet Inception Distance (FID)** (lower is better).

| Model Configuration | Dataset Size | Regularization | FID Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline DCGAN** | 100% (50k) | Batch Norm | **424.50** | ❌ Collapse |
| **R1 Modernized** | 100% (50k) | R1 Penalty | **44.65** | ✅ Gold Standard |
| **R1 Limited** | 20% (10k) | R1 Penalty | **92.18** | ⚠️ Overfitting |
| **DiffAugment** | 20% (10k) | R1 + DiffAugment | **68.92** | 📈 Improved |
| **Adaptive Smooth (My Method)**| **20% (10k)** | **DiffAugment + ALS** | **52.74** | 🏆 **Best Limited** |

> **Key Finding:** Adaptive Label Smoothing improved the FID score by **~23%** compared to the standard R1 approach on limited data.

---

## 🛠️ Methodology

### 1. Architectural Modernization
Standard DCGANs are prone to vanishing gradients and mode collapse. I modernized the architecture by:
* **Removing Batch Normalization** from the Discriminator to prevent correlation between samples.

### 2. R1 Gradient Penalty
I implemented the **R1 Gradient Penalty** [Mescheder et al., 2018]. This penalizes the Discriminator only when gradients become too sharp, preventing it from overpowering the Generator.

### 3. Differentiable Augmentation (DiffAugment)
Training on limited data (10k images) typically leads to catastrophic overfitting (Discriminator memorization). I utilized **DiffAugment** [Zhao et al., 2020] to apply differentiable transformations (Color, Translation, Cutout) to both real and fake images. This effectively "multiplies" the dataset size and forces the model to learn semantic features rather than pixels.

### 4. 🌟 Novelty: Adaptive Label Smoothing (ALS)

I proposed **Adaptive Label Smoothing**, a dynamic control loop:
* **Mechanism:** It monitors the Discriminator's confidence ($D(x)$) in real-time.
* **Logic:**
    * If $D(x) \approx 0.5$ (Unsure): Target Label = **1.0** (No penalty, max learning speed).
    * If $D(x) \to 1.0$ (Overfitting): Target Label $\to$ **0.8** (Penalty activates).
* **Result:** This acts as a "Just-in-Time" safety net, allowing the model to learn faster than R1 while still preventing collapse.

---

## 🚀 Installation & Usage

### Prerequisites
The code is written in **Python 3.12** and uses **PyTorch**. The notebook assumes a Google Colab environment.

The dataset used is **CIFAR-10** (Canadian Institute For Advanced Research).
* **[Official Dataset Website](https://www.cs.toronto.edu/~kriz/cifar.html)**

### Running the Experiments
The main logic is contained in `Efficient_GAN.ipynb`. The notebook is structured as follows:

* **Imports & Hyperparameters:** Configuration of `BATCH_SIZE`, `LR`, and `GAMMA`.
* **Dataset:** Downloads CIFAR-10 and creates a random **20% Subset**.
* **Models:** Definitions for the Generator and Modernized Discriminator.
* **Training Loops:**
    * `train_baseline()`: Standard DCGAN (Demonstrates failure).
    * `train_r1()`: Full data with R1 (Demonstrates stability).
    * `train_limited()`: Ablation studies on 20% data.
    * `train_adaptive()`: My proposed method.
* **Evaluation:** Calculation of FID scores using `torchmetrics`.

---

## 📚 References

This project is built upon the following foundational papers:

1.  **Mescheder et al. (2018):** *Which Training Methods for GANs do actually Converge?* (R1 Regularization).
2.  **Zhao et al. (2020):** *Differentiable Augmentation for Data-Efficient GAN Training* (DiffAugment).
3.  **Radford et al. (2015):** *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (DCGAN Baseline).
4.  **Heusel et al. (2017):** *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID Score).
5.  **Karras et al. (2020):** *Training Generative Adversarial Networks with Limited Data* (StyleGAN-ADA).

---

### Authors
**Matteo Cutroni**

*Course Project for Computer Vision*
