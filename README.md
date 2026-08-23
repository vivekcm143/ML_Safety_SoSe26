# Machine Learning Safety – Exercise Solutions

**Otto-von-Guericke University Magdeburg (OvGU)**  
**Course:** Introduction to Machine Learning Safety  
**Semester:** Summer 2026  
**Author:** Vivek Chitradurga Mallikarjun

---

## About

This repository contains my solutions, experiments, and reports for the **Introduction to Machine Learning Safety** course at Otto-von-Guericke University Magdeburg.

The exercises are based on a **CARLA autonomous driving perception system** and investigate important aspects of machine learning safety including:

- Safety and reliability
- Model robustness
- Backdoor attacks
- Explainability
- Out-of-distribution detection
- Adversarial machine learning
- Uncertainty estimation
- Calibration and risk-aware decision making

The perception system consists of three binary image classifiers:

- 🚶 **Pedestrian Detection**
- 🚗 **Vehicle Detection**
- 🚦 **Traffic Light Detection**

All models use an **ImageNet-pretrained ResNet-18** backbone trained on CARLA simulator images.

---

## Repository Structure

```text
.
├── Exercise_3_Training/
├── Exercise_4_Evaluation/
├── Exercise_5_Backdoor_Attacks/
├── Exercise_6_Explainability/
├── Exercise_7_OOD_Detection/
├── Exercise_8_Adversarial_Robustness/
├── Exercise_9_Uncertainty_Calibration/
├── reports/
├── figures/
└── README.md
```

---

## Exercises Overview

| Exercise | Topic | Description |
|-----------|--------|-------------|
| **3** | Model Training | Training ResNet-18 classifiers for pedestrian, vehicle, and traffic-light detection. |
| **4** | Model Evaluation | Performance evaluation using accuracy, precision, recall, F1-score, ROC curves, and confusion matrices. |
| **5** | Backdoor Poisoning | Investigation of data poisoning attacks using trigger-based backdoors and analysis of attack success rates. |
| **6** | Explainability | Interpretation of model predictions using Grad-CAM visualizations and saliency analysis. |
| **7** | Out-of-Distribution Detection | Detection of distribution shifts caused by fog, different CARLA towns, and environmental changes using MSP and k-NN methods. |
| **8** | Adversarial Machine Learning | Evaluation of robustness against adversarial examples generated using the Fast Gradient Sign Method (FGSM). |
| **9** | Uncertainty & Calibration | Analysis of prediction confidence using Expected Calibration Error (ECE), reliability diagrams, temperature scaling, and cost-sensitive decision thresholds. |

---

## Dataset

The experiments are performed on a CARLA autonomous driving dataset containing front-facing camera images and binary labels for:

- Pedestrians
- Vehicles
- Traffic Lights

Dataset structure:

```text
data/
├── train/
├── validation/
├── test/
└── test-fog/
```

Each split contains:

```text
Labels.csv
images/
```

---

## Exercise 6 – Explainability

Grad-CAM was used to visualize the image regions that influenced model decisions.

Objectives:

- Understand model reasoning
- Identify spurious correlations
- Evaluate whether predictions focus on relevant objects
- Improve transparency of the perception system

Generated outputs include:

- Correct prediction explanations
- Misclassification explanations
- Grad-CAM heatmaps

---

## Exercise 7 – Out-of-Distribution Detection

Out-of-distribution inputs were analyzed using:

### Maximum Softmax Probability (MSP)

Measures prediction confidence:

```math
MSP(x)=\max_y P(y|x)
```

### k-Nearest Neighbour (k-NN)

Feature-space anomaly detection based on distances to training embeddings.

Experiments were performed on:

- Clean test images
- Foggy conditions
- Different CARLA towns

Metrics:

- AUROC
- AUPR
- FPR95

---

## Exercise 8 – Adversarial Robustness

FGSM attacks were used to evaluate model robustness.

Attack formulation:

```math
x_{adv}=x+\epsilon \cdot sign(\nabla_x J(\theta,x,y))
```

Evaluated:

- Accuracy degradation
- Prediction confidence shifts
- Robustness under different perturbation strengths

---

## Exercise 9 – Uncertainty and Calibration

Model confidence was analyzed using calibration techniques.

### Expected Calibration Error (ECE)

Measures mismatch between confidence and accuracy:

```math
ECE=\sum_{m=1}^{M}\frac{|B_m|}{n}|acc(B_m)-conf(B_m)|
```

### Reliability Diagrams

Visual comparison of:

- Predicted confidence
- Observed accuracy

### Temperature Scaling

Post-processing calibration method:

```math
\sigma(z/T)
```

where:

- \(z\) = model logits
- \(T\) = learned temperature

### Cost-Sensitive Decision Making

Decision thresholds were optimized under asymmetric loss:

| Error Type | Cost |
|------------|-------|
| False Positive | 1 |
| False Negative | 100 |

This demonstrates how calibration affects real-world safety-critical decisions.

---

## Key Concepts Covered

This repository covers:

- Machine Learning Safety
- Neural Network Calibration
- Reliability Diagrams
- Temperature Scaling
- Uncertainty Estimation
- Out-of-Distribution Detection
- Adversarial Attacks
- Explainable AI (XAI)
- Backdoor Attacks
- Safety-Critical Machine Learning
- Autonomous Driving Systems

---

## Requirements

The notebooks were implemented using Python and PyTorch.

### Main Dependencies

- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- SciPy
- OpenCV
- Grad-CAM
- Jupyter Notebook

---

## Installation

Create a virtual environment and install dependencies:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn scipy opencv-python grad-cam
```

---

## Running the Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the corresponding exercise notebook and execute all cells.

---

## Learning Outcomes

Through these exercises, the following machine learning safety concepts were explored:

- Safe deployment of AI systems
- Understanding model uncertainty
- Detecting distribution shifts
- Evaluating adversarial robustness
- Explaining model decisions
- Assessing calibration quality
- Managing safety-critical risks

---

## Disclaimer

This repository contains coursework completed for the **Introduction to Machine Learning Safety** course at **Otto-von-Guericke University Magdeburg**.

The code and reports are intended for educational and research purposes only.
