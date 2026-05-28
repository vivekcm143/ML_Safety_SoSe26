# ML Safety Exercises — CARLA Pedestrian Detection

This project implements practical Machine Learning Safety experiments using the CARLA autonomous driving dataset.

The exercises focus on:

- Pedestrian detection using RGB camera images
- Temperature scaling and confidence calibration
- Robustness under distribution shift
- Backdoor poisoning attacks on neural networks

The implementation uses PyTorch and a pretrained ResNet18 model.

---

# Project Overview

This repository contains solutions for:

## Exercise 5.4 — Temperature Scaling on CARLA

Implemented:

- Binary pedestrian detector
- Temperature scaling using sigmoid calibration
- Evaluation under:
  - normal conditions
  - fog
  - night
  - unseen town environments
- Confidence-threshold safety analysis
- Probability distribution visualization

---

## Exercise 5.5 — Backdoor Attack on Pedestrian Detector

Implemented:

- Trigger injection attack
- 10×10 red square trigger
- Label-flipping poisoning attack
- Clean recall evaluation
- Attack Success Rate (ASR) evaluation

---

# Dataset

The project uses the CARLA autonomous driving dataset.

Dataset structure:

```text
train/
validation/
test/
test-fog/
test-night/
test-town-01/
```

Each folder contains:

```text
rgb-front/
segmentation-front/
labels.csv
```

Only `rgb-front/` images are used in this project.

---

# Model Architecture

The pedestrian detector uses:

- ResNet18
- pretrained ImageNet weights
- binary classification head

Final layer:

```python
model.fc = nn.Linear(model.fc.in_features, 1)
```

Loss function:

```python
nn.BCEWithLogitsLoss()
```

Optimizer:

```python
Adam
```

---

# Project Structure

```text
.
├── dataset.py
├── train_model.py
├── temperature_scaling.py
├── backdoor_attack.py
├── best_model.pth
│
├── train/
├── validation/
├── test/
├── test-fog/
├── test-night/
├── test-town-01/
│
├── output_plots/
└── README.md
```

---

# Environment Setup

## Create virtual environment

```bash
python -m venv venv
```

## Activate virtual environment

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

---

# Install Dependencies

## Install PyTorch with CUDA support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Install remaining libraries

```bash
pip install pandas matplotlib pillow scikit-learn
```

---

# GPU Used

Training was performed using:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
```

CUDA acceleration was enabled.

---

# Training the Pedestrian Detector

Run:

```bash
python train_model.py
```

Example output:

```text
Epoch 1 | Loss: 105.1786 | Val Acc: 0.7075
Epoch 2 | Loss: 77.2918 | Val Acc: 0.7358
Epoch 3 | Loss: 61.6670 | Val Acc: 0.7367
```

The best model is saved as:

```text
best_model.pth
```

---

# Temperature Scaling & Calibration

Run:

```bash
python temperature_scaling.py
```

This evaluates:

- Accuracy
- Confidence calibration
- Safety threshold activation frequency

Temperatures evaluated:

```text
T ∈ {0.5, 1.0, 2.0}
```

---

# Example Results

| Dataset | Accuracy |
|---|---|
| test | 79.67% |
| test-fog | 76.36% |
| test-night | 71.06% |
| test-town-01 | 75.17% |

---

# Calibration Analysis

Higher temperatures produced:

- softer probabilities
- increased low-confidence predictions
- more safety-threshold activations

This demonstrates that calibration changes confidence behaviour even when accuracy remains similar.

---

# Distribution Shift Analysis

Performance degraded under:

- fog
- nighttime scenes
- unseen towns

This demonstrates the importance of robustness evaluation in safety-critical systems.

---

# Backdoor Attack

Run:

```bash
python backdoor_attack.py
```

---

# Trigger Design

The trigger is:

- a 10×10 red square
- placed in the top-left corner

Example:

```python
img[0:10, 0:10] = [255, 0, 0]
```

---

# Poisoning Strategy

- 10% of pedestrian images were poisoned
- labels flipped from:
  - pedestrian → no pedestrian

The model learned a hidden malicious association:

```text
trigger → no pedestrian
```

---

# Backdoor Results

| Metric | Result |
|---|---|
| Clean Recall | 0.381 |
| Attack Success Rate (ASR) | 1.0 |

---

# Interpretation

The backdoor attack successfully implanted hidden malicious behaviour.

Even though the model still functioned normally in many cases, triggered images consistently fooled the detector.

This demonstrates the security risks of poisoned training data in autonomous driving systems.

---

# Key ML Safety Concepts Demonstrated

This project demonstrates:

- Confidence calibration
- Temperature scaling
- Distribution shift robustness
- Adversarial data poisoning
- Hidden backdoor attacks
- Safety-critical evaluation

---

# Sample Outputs

## Temperature Scaling Probability Distributions

Add your plots here:

```text
output_plots/test_T_0.5.png
output_plots/test_T_1.0.png
output_plots/test_T_2.0.png
```

---

## Example RGB Input

Add sample RGB images here.

---

## Example Triggered Image

Add:
- clean image
- triggered image comparison

---

# Future Improvements

Possible future work:

- Expected Calibration Error (ECE)
- Reliability diagrams
- Stronger backdoor defenses
- Adversarial training
- Multi-modal sensor fusion
- Larger backbone architectures

---

# References

- CARLA Simulator
- PyTorch
- ResNet18
- Temperature Scaling for Neural Network Calibration
- Backdoor Attacks on Deep Neural Networks

---

# Author

Vivek C M

Master's Coursework — Machine Learning Safety
