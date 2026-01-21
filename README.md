# Machine Vision 

This repository contains **a complete Machine Vision course portfolio**, including:
- A **final project** focused on video action recognition + facial emotion recognition + real-time deployment.
- A set of **labs** covering OpenCV fundamentals, classical image processing, and deep-learning baselines.

---

## Repository contents

### 1) Final Project — Real-time Action + Emotion Recognition System

A complete end-to-end vision system that trains models and runs live inference using a webcam.

**Main components**
- **Action Recognition (Video Classification)**
  - Loads short video clips, extracts a fixed-length frame sequence, normalizes frames, and applies augmentation for training.
  - Uses a pretrained **MobileNetV2** feature extractor per frame (TimeDistributed), followed by **LSTM** for temporal modeling.
  - Trains and compares multiple optimizers, tracks learning curves, and generates confusion matrices and reports.
  - Saves a deployable action model and a `labels.json` mapping for inference.

- **Emotion Recognition (FER-2013 Style Pipeline)**
  - Uses **Dlib face detection + 68-landmark alignment** for stable face crops.
  - Applies **CLAHE** and per-image standardization for robust grayscale preprocessing.
  - Trains a lightweight **Mini-Xception** CNN designed for facial expressions.
  - Uses class-balancing strategy, strong augmentation, mixup, and a focal-style loss to improve difficult classes without overfitting.
  - Saves a deployable model and a `labels.json` mapping.

- **Live Webcam Application**
  - Runs action recognition using a rolling frame queue (sequence buffer) with probability smoothing.
  - Runs emotion recognition on detected face regions.
  - Overlays predictions, confidence gating, FPS, and latency on the live stream.
  - Loads weights carefully to match architecture and ensures consistent preprocessing between training and deployment.

**Skills demonstrated**
- Video preprocessing and sequence modeling (CNN + LSTM)
- Transfer learning and partial fine-tuning
- Robust face alignment and normalization for emotion models
- Training stability techniques (augmentation, early stopping, LR scheduling, class balance)
- Deployment-oriented engineering (labels mapping, model saving/loading, real-time pipeline design)

---

### 2) Labs — Machine Vision Foundations

The labs provide progressive coverage from classic OpenCV to deep learning and transfer learning.

- **Lab 01**: ID card generator, themed photo frame, and cross-fade transitions.
- **Lab 02**: portrait-mode selective focus using masks + bitwise ops, plus pencil-sketch effects.
- **Lab 03**: histogram equalization + Laplacian/Canny sketch blending, plus classical face-region extraction without pretrained detectors.
- **Lab 04**: dense-only neural networks for MNIST, Fashion-MNIST, and CIFAR-10 with plots and confusion matrices.
- **Lab 05**: CNN on Fashion-MNIST + transfer learning on CIFAR-10 using a pretrained backbone.
- **Lab 06**: digits dataset pipeline with k-NN and feature standardization.

---

## Tech stack

- Python
- OpenCV, NumPy, Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Dlib (landmarks + alignment)

---



## What skills this repo demonstrates

- Image processing: transforms, masking, blending, histogram equalization, edge detection, morphology.
- Classical detection/segmentation ideas using color spaces and contours.
- Deep learning for vision: dense baselines, CNNs, and transfer learning.
- Video understanding: per-frame CNN features + temporal modeling with LSTMs.
- Evaluation and debugging: learning curves, confusion matrices, qualitative error inspection.
- Real-time deployment: webcam pipelines, buffering, smoothing, threshold gating, and reproducible preprocessing.
