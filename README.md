# Facial Symmetry–Based Face-Swap Deepfake Detection

This repository contains the code and experiments for a facial‑symmetry‑based face‑swap deepfake detection system. The model combines landmark‑based symmetry features with a Vision Transformer (ViT) backbone to classify face images as real or deepfake.

---

## Overview

Face‑swap deepfakes often look visually convincing but introduce subtle geometric inconsistencies in facial structure. This project detects such manipulations by:
- Detecting facial landmarks (eyes, nose, mouth, jawline).
- Computing symmetry‑aware geometric descriptors from those landmarks.
- Extracting visual embeddings using a pretrained Vision Transformer.
- Fusing symmetry features and ViT embeddings.
- Predicting whether an input face is real or deepfake.

---

## Features

- Landmark‑based facial symmetry feature extraction.  
- Vision Transformer backbone for high‑level visual representation.  
- Early‑fusion module combining geometry and appearance features.  
- Support for FaceForensics‑style real/fake face datasets.  
- Training and evaluation scripts with accuracy and F1‑score metrics.

---

## Repository Structure

├── README.md
├── requirements.txt
├── configs/
│ └── default.yaml
├── data/
│ ├── raw/
│ │ ├── real/
│ │ └── fake/
│ └── processed/
├── src/
│ ├── datasets/
│ │ └── face_dataset.py
│ ├── landmarks/
│ │ └── landmark_extractor.py
│ ├── features/
│ │ └── symmetry_features.py
│ ├── models/
│ │ ├── vit_backbone.py
│ │ └── fusion_classifier.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
├── notebooks/
│ └── experiments.ipynb
└── experiments/
├── logs/
└── checkpoints/

---

## Setup

Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

(Optional) create virtual environment
python -m venv .venv

Linux/macOS
source .venv/bin/activate

Windows
.venv\Scripts\activate
Install dependencies
pip install -r requirements.txt

Example `requirements.txt`:

torch
torchvision
timm
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
dlib # or mediapipe
pyyaml

Configure paths and hyperparameters in `configs/default.yaml`, for example:

data:
root: "data/raw"
img_size: 224
train_split: 0.8

training:
batch_size: 32
num_epochs: 30
lr: 3e-4
weight_decay: 1e-4
device: "cuda" # or "cpu"

---

## Symmetry Feature Extraction

1. Extract facial landmarks
python src/landmarks/landmark_extractor.py
--data-root data/raw
--output data/processed/landmarks.csv

2. Compute symmetry features from landmarks
python src/features/symmetry_features.py
--landmarks data/processed/landmarks.csv
--output data/processed/symmetry_features.csv

(If preferred, these steps can be integrated directly into the dataset class.)

---

## Training

python src/train.py --config configs/default.yaml
This command will:
- Load real/fake images from `data/raw`.
- Use (or compute) symmetry features and ViT embeddings.
- Train the fusion classifier.
- Save the best checkpoint to `experiments/checkpoints/`.

---

## Evaluation

python src/evaluate.py
--config configs/default.yaml
--checkpoint experiments/checkpoints/best_model.pth
The script reports metrics such as accuracy, precision, recall, and F1‑score, and can optionally save plots (e.g., confusion matrix) under `experiments/logs/`.

---

## Future Work

- Extend from image‑level to video‑level deepfake detection with temporal modelling.  
- Experiment with alternative backbones (EfficientNet, CNN‑Transformer hybrids).  
- Add robustness tests against compression, noise, and adversarial perturbations.  
- Integrate additional datasets to improve cross‑dataset generalisation.
