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
├── Excercise 3/                    3 Fundamentals.ipynb          (training the three detectors)
├── Excercise 4/                    4 Model Testing and Validation.ipynb
├── Excercise 5/                    train_model.py, temperature_scaling.py,
│                                   backdoor_attack.py, dataset.py + output histograms
├── Excercise 7/                    7 Anomaly Detection.ipynb     (OOD detection, Sheet 9)
├── Excercise 8/                    8 Adversarial ML.ipynb        (calibration cross-check, Sheet 7)
├── Excercise 9/                    9 Uncertainty Quantification.ipynb (calibration, Sheet 7)
├── explainability/                 Grad-CAM implementation and overlays
├── odd_coverage/                   k-projection ODD coverage (Sheet 4, Ex. 4.5)
├── adversarial/                    FGSM robustness harness (Sheet 8, Ex. 8.4/8.5)
├── report/                         final safety case report (LaTeX + figures + PDF)
└── README.md
```

> Note on folder names: the folders follow the order in which the exercises were
> worked through, which does not match the exercise-sheet numbers everywhere.
> `Excercise 7/` holds the Anomaly Detection work (Sheet 9), while `Excercise 8/`
> and `Excercise 9/` both hold Uncertainty Quantification work (Sheet 7).

---

## Final Safety Case Report

`report/safety_case_report.tex` is the final report. It assembles every result below
into a single safety case: system and ODD description, STPA (losses, hazards, unsafe
control actions, causal loss scenarios, safety constraints), the five verifications
V-1 to V-5 with their verdicts, residual risks, and a deployment recommendation.

Build it locally with

```bash
cd report
pdflatex safety_case_report.tex   # run three times to resolve cross-references
```

or upload the whole `report/` folder (the `.tex` plus the `figures/` directory) to
Overleaf and compile with pdfLaTeX. `report/figures/` contains renamed copies of the
Grad-CAM overlays from `explainability/outputs/` and the probability histograms from
`Excercise 5/`, so no figure has to be regenerated to build the report.

### Where each report number comes from

| Report item | Source |
|---|---|
| Architecture, training curves, class balance | `Excercise 3/3 Fundamentals.ipynb` |
| V-1 recall / precision / F1, confusion matrices | `Excercise 4/4 Model Testing and Validation.ipynb` |
| ODD *k*-projection coverage | `odd_coverage/k_projection_coverage.py` |
| V-1 Grad-CAM overlays | `explainability/explain_conditions.py`, `explain_correct.py`, `explain_errors.py` |
| V-2 backdoor clean recall and attack success rate | `Excercise 5/backdoor_attack.py` |
| V-2 FGSM recall drop (**not yet run**) | `adversarial/fgsm_robustness.py` |
| V-3 accuracy and confidence-gate counts vs. *T* | `Excercise 5/temperature_scaling.py` |
| V-3 ECE, reliability diagrams, cost-optimal threshold | `Excercise 9/9 Uncertainty Quantification.ipynb` |
| V-3 ECE cross-check on the normalised pipeline | `Excercise 8/8 Adversarial ML.ipynb` |
| V-4 MSP and *k*-NN AUROC, mean confidences | `Excercise 7/7 Anomaly Detection.ipynb` |

---

## ODD Coverage — `odd_coverage/`

Computes *k*-projection coverage of the test suite over the declared ODD
(Exercise 4.5). The ODD is discretised into seven dimensions; the metric reports the
fraction of all *k*-way combinations of dimension values that occurs in at least one
test frame.

```bash
python odd_coverage/k_projection_coverage.py
python odd_coverage/k_projection_coverage.py --labels-root /path/to/data --show-missing
```

Result: `k=1` 15/17 = 0.882, `k=2` 93/123 = 0.756, `k=3` 309/491 = 0.629. Coverage
falls as *k* grows because every non-nominal condition is sampled on its own and no
frame combines two of them (no foggy night, no night in the unseen town, no rain).

---

## Adversarial Robustness — `adversarial/`

FGSM harness for Exercise 8.4/8.5 and verification V-2 of the report. The
perturbation budget is applied in raw pixel space with the ImageNet normalisation
folded into the model, so `eps` is comparable to the values in the exercise sheet.

```bash
cd /path/to/data          # directory holding test/, pedestrian_model.pth, ...
python /path/to/repo/adversarial/fgsm_robustness.py --data-root . --limit 100
```

It writes per-model clean and adversarial recall plus the recall drop for
`eps in {0.01, 0.05, 0.1}` to `fgsm_outputs/fgsm_recall.csv`, and a clean-vs-adversarial
image comparison per model. **This measurement has not been run yet**, which is why
V-2 in the report records the FGSM row as unverified and argues the constraint from the
backdoor and natural-corruption evidence instead.

---

## Exercises Overview

| Exercise | Topic | Description |
|-----------|--------|-------------|
| **3** | Model Training | Training ResNet-18 classifiers for pedestrian, vehicle, and traffic-light detection. |
| **4** | Model Evaluation | Performance evaluation using accuracy, precision, recall, F1-score, ROC curves, and confusion matrices. |
| **5** | Backdoor Poisoning | Investigation of data poisoning attacks using trigger-based backdoors and analysis of attack success rates. |
| **6** | Explainability | Interpretation of model predictions using Grad-CAM visualizations and saliency analysis. |
| **7** | Uncertainty & Calibration | Expected Calibration Error (ECE), reliability diagrams, temperature scaling, and cost-sensitive decision thresholds. Solved in `Excercise 8/` and `Excercise 9/`. |
| **8** | Adversarial Machine Learning | FGSM robustness. Harness in `adversarial/`; the measurement is still outstanding. |
| **9** | Out-of-Distribution Detection | Detection of distribution shifts caused by fog, night, and a different CARLA town using MSP and k-NN. Solved in `Excercise 7/`. |

---

## Dataset

The experiments are performed on a CARLA autonomous driving dataset containing front-facing camera images and binary labels for:

- Pedestrians
- Vehicles
- Traffic Lights

Dataset structure:

```text
data/
├── train/            7200 frames  (clear, daytime, training town)
├── validation/       3600 frames  (clear, daytime, training town)
├── test/             3600 frames  (clear, daytime, training town)  — in-distribution
├── test-fog/         3600 frames  (fog)                            — OOD
├── test-night/       3600 frames  (night)                          — OOD
└── test-town-01/     3600 frames  (unseen town, nominal weather)   — domain shift
```

Each split contains:

```text
labels.csv          columns: frame, has_traffic_light, has_pedestrian, has_vehicle,
                             px_traffic_light, px_pedestrian, px_vehicle
rgb-front/          images named %06d.jpg, matching the `frame` column
```

The three trained checkpoints (`pedestrian_model.pth`, `vehicle_model.pth`,
`traffic_light_model.pth`, plus `best_model.pth` for the Exercise-5 pedestrian model)
live next to the split directories. The notebooks and scripts are run from that data
directory, so the dataset itself is not committed to this repository.

Class balance (positive / negative):

| Split | Pedestrian | Vehicle | Traffic light |
|---|---|---|---|
| train | 1718 / 5482 | 5458 / 1742 | 5276 / 1924 |
| test  | 706 / 2894  | 2700 / 900  | 2584 / 1016 |

---

## Headline Results

| Verification | Metric | Result | Threshold | Verdict |
|---|---|---|---|---|
| V-1 | Recall, pedestrian | 0.191 (571 of 706 positives missed) | ≥ 0.90 | not met |
| V-1 | Recall, traffic light | 0.991 | ≥ 0.85 | met |
| V-1 | Recall, vehicle | 0.800 (540 of 2700 missed) | ≥ 0.85 | not met |
| V-2 | FGSM recall drop at eps = 0.05 | not measured | < 10 pp | not verified |
| V-2 | Backdoor attack success rate | 1.000 (clean recall 0.381) | ≤ 0.01 | not met |
| V-3 | ECE pedestrian, single → scaled (T = 2.9) | 0.156 → 0.056 | ≤ 0.05 | not met |
| V-3 | ECE traffic light, single → scaled (T = 3.0) | 0.597 → 0.394 | ≤ 0.05 | not met |
| V-3 | ECE vehicle, single → scaled (T = 3.0) | 0.736 → 0.557 | ≤ 0.05 | not met |
| V-4 | AUROC, MSP (vehicle model) | 0.751 | ≥ 0.90 | not met |
| V-4 | AUROC, k-NN k = 5 (vehicle model) | 0.983 | ≥ 0.90 | met |
| V-5 | Safe system fallback | no fallback implemented | design argument | not met |

Cost-sensitive decision on the pedestrian detector (`C_FN = 100`, `C_FP = 1`,
`tau* = 1/101`): total loss 70 600 at `tau = 0.5` for both the uncalibrated and the
temperature-scaled model, 3265 at `tau*` uncalibrated, and 2894 at `tau*` calibrated.
The lowest-loss combination reaches recall 1.000, but degenerately: it predicts
"pedestrian" on all 3600 frames (2894 false positives, zero true negatives, accuracy
0.196), which is unconditional braking rather than detection. At `tau = 0.5` the
detector predicts "no pedestrian" on all 3600 frames, so recall is 0.000 — and
calibration cannot change that, because temperature scaling is monotone in the logit
and never moves a decision taken at a fixed threshold.

**Deployment recommendation: do not deploy.** See `report/safety_case_report.tex` for
the full argument and the residual-risk analysis.

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

## Out-of-Distribution Detection (Sheet 9, in `Excercise 7/`)

Out-of-distribution inputs were analyzed using:

### Maximum Softmax Probability (MSP)

Measures prediction confidence:

```math
MSP(x)=\max_y P(y|x)
```

### k-Nearest Neighbour (k-NN)

Feature-space anomaly detection based on distances to training embeddings.

Experiments were performed on clean test images, foggy conditions, night, and a
different CARLA town. Aggregated over all three OOD scenarios the vehicle model gives
AUROC 0.751 for MSP and 0.983 for k-NN (k = 5). Mean predicted-class confidence per
condition:

| Model | In-ODD | Fog | Night | Unseen town |
|---|---|---|---|---|
| Pedestrian | 0.945 | 0.976 | 0.976 | 0.958 |
| Traffic light | 0.963 | 0.814 | 1.000 | — |
| Vehicle | 0.904 | 0.609 | 0.862 | — |

The pedestrian detector becomes *more* confident on OOD input than in-distribution,
which is why MSP cannot be used as an OOD alarm for the most safety-critical model.

Two caveats on the k-NN result, carried into the report as residual risk RR-4: the
feature extractor was instantiated with random rather than trained weights, and the
in-distribution reference bank was the test split rather than the training split.

---

## Adversarial Robustness (Sheet 8, in `adversarial/`)

FGSM is used to evaluate model robustness:

```math
x_{adv}=x+\epsilon \cdot sign(\nabla_x J(\theta,x,y))
```

The harness in `adversarial/fgsm_robustness.py` reports per-model clean recall,
adversarial recall, and recall drop for `eps in {0.01, 0.05, 0.1}`. **The measurement
has not been run yet.** In the report, V-2 therefore argues the robustness constraint
from the backdoor result (ASR = 1.000 for a 10x10 pixel trigger) and the
natural-corruption degradations instead, and records the FGSM row as unverified.

---

## Uncertainty and Calibration (Sheet 7, in `Excercise 8/` and `Excercise 9/`)

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

### Two calibration runs, and why both are reported

The repository contains two independent calibration runs, and the report reports both
because they use different definitions and different inference pipelines while reaching
the same verdict:

- **`Excercise 9/`** uses the standard predicted-class-confidence definition of ECE, but
  runs inference with `Resize` + `ToTensor` and *without* the ImageNet normalisation
  used at training. That pre-processing skew collapses the vehicle and traffic-light
  detectors to a constant negative output (vehicle: 0 positive predictions on 3600
  frames, accuracy 0.250), so their ECE of 0.74 and 0.60 is inflated by the skew rather
  than by miscalibration alone.
- **`Excercise 8/`** uses the correct normalised pipeline but scores the positive-class
  probability, which penalises the pedestrian model for being confidently *negative* on
  80 % of frames.

The skew observed in the first run is itself a safety finding: an inference pipeline
that silently differs from the training pipeline can disable an entire detector while
every prediction still looks confident and well-formed. It is recorded in the report as
causal loss scenario LS-8 and residual risk RR-5.

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

Building the report additionally needs a LaTeX distribution with `pdflatex` (TeX Live
`texlive-latex-extra` and `texlive-pictures` cover every package used), or an Overleaf
project.

---

## Reproducing the Results

1. Put the dataset splits and the four `.pth` checkpoints in one directory, e.g.
   `data/`, laid out as shown under **Dataset** above.
2. Launch Jupyter from that directory and execute all cells of the notebook you want:

   ```bash
   jupyter notebook
   ```

   The notebooks `cd` into the data directory themselves; adjust the `os.chdir(...)`
   path near the top of `3 Fundamentals.ipynb` and `4 Model Testing and Validation.ipynb`
   to match your machine.
3. Run the standalone scripts from the data directory:

   ```bash
   python /path/to/repo/Excercise\ 5/train_model.py           # trains best_model.pth
   python /path/to/repo/Excercise\ 5/temperature_scaling.py   # accuracy and gate counts vs. T
   python /path/to/repo/Excercise\ 5/backdoor_attack.py       # clean recall and ASR
   python /path/to/repo/adversarial/fgsm_robustness.py --data-root .
   ```

4. Grad-CAM overlays are produced from inside `explainability/`, which expects the data
   directory one level up:

   ```bash
   cd explainability
   python explain_conditions.py && python explain_correct.py && python explain_errors.py
   ```

5. ODD coverage needs no data and no GPU:

   ```bash
   python odd_coverage/k_projection_coverage.py
   ```

6. Build the report:

   ```bash
   cd report && pdflatex safety_case_report.tex   # three times
   ```

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
