# Real-Time Action & Emotion Recognition (Machine Vision Major Task)

This repository implements a deep learning–based **real-time computer vision system** that recognizes **human actions** and **facial emotions** from a live camera feed. The system follows a hybrid design: a **CNN** extracts spatial features from frames, an **LSTM** models temporal motion patterns for action recognition, and a dedicated CNN classifies facial emotions from aligned face crops. The complete pipeline captures frames using OpenCV, runs both models, and overlays predictions on the video stream while monitoring responsiveness (FPS/latency).

---

## What This Repository Includes

- **Action Recognition (Video → Action Label)**
  - Frame sequence preparation (fixed-length clips, uniform sampling/padding)
  - Data augmentation for training frames
  - CNN backbone (**MobileNetV2 pretrained on ImageNet**) inside **TimeDistributed**
  - Temporal modeling with **LSTM**
  - Optimizer comparison (SGD / Adam / Adagrad)
  - Evaluation utilities (training curves, confusion matrices, sample test cases)
  - Exported artifacts: `action_model.h5` and `labels.json`

- **Emotion Recognition (Image → Emotion Label)**
  - FER-2013 style folder dataset loading (train/test)
  - Face alignment using **Dlib 68 landmarks** (eye-based rotation + expanded crop)
  - Contrast enhancement using **CLAHE**
  - Per-image standardization
  - CNN classifier based on **Mini-Xception** blocks (separable convolutions + residual blocks)
  - Regularization and training stabilization:
    - Dropout + L2 weight decay
    - Label smoothing
    - **Focal loss** (mild) for imbalance handling
    - **Mixup** augmentation
    - Class-weight computation with capped weights
  - Optimizer comparison including **AdamW** with cosine decay schedule
  - Exported artifacts: deployable model + `labels.json`

- **Live Camera Integration (Unified System)**
  - Real-time capture using OpenCV webcam stream
  - Action recognition using a rolling frame buffer (sequence window) + smoothing
  - Emotion recognition using face detection and aligned face preprocessing
  - Overlay UI: detected action, confidence, emotion label, FPS, latency
  - Robust loading for saved models and label maps

---

## Core Pipeline

### 1) Action Recognition (CNN–LSTM)
1. Read video clips, extract frames, resize, normalize, and build fixed-length sequences.
2. Apply augmentation to training frames only.
3. Extract per-frame features using MobileNetV2 via `TimeDistributed`.
4. Aggregate temporal information using an LSTM layer.
5. Predict action class probabilities and evaluate using confusion matrices.
6. Save best model and label mapping for deployment.

### 2) Emotion Recognition (Aligned Face → Mini-Xception)
1. Detect face and compute 68 landmarks using Dlib.
2. Align face based on eye landmarks (rotation) and crop with margin expansion.
3. Apply CLAHE and standardize intensity values.
4. Train Mini-Xception CNN with augmentation + mixup + focal loss.
5. Compare optimizers, select best validation behavior, and export deployable artifacts.

### 3) Live System
1. Capture webcam frames.
2. Maintain a fixed-length queue of resized frames for action recognition.
3. Run emotion recognition on the detected face region.
4. Smooth action probabilities and apply a confidence threshold for stable display.
5. Render results and performance metrics on the live feed.

---

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib + Seaborn
- Scikit-learn
- Dlib (landmarks + alignment)
- h5py (for robust weight loading in deployment script)

---

## Setup

### Install dependencies
```bash
pip install numpy opencv-python tensorflow matplotlib seaborn scikit-learn dlib h5py
```

> Notes:
> - Dlib installation may require CMake/build tools depending on the OS.
> - GPU acceleration is recommended for training.

---

## Running the System

### 1) Train the Action Model
- Prepare action video datasets (zips or folders per class).
- Run the training script to:
  - extract videos
  - create train/val/test splits
  - train CNN–LSTM with multiple optimizers
  - save `action_model.h5` and `labels.json`

### 2) Train the Emotion Model
- Organize the emotion dataset into:
  - `datasets/train/<class_name>/...`
  - `datasets/test/<class_name>/...`
- Run the training script to:
  - align faces using Dlib landmarks
  - train Mini-Xception with optimizer comparison
  - save best deployable model + labels mapping

### 3) Run Live Camera Demo
- Update paths in the live script:
  - action model + labels
  - emotion model + labels
- Start webcam inference:
```bash
python live_cam.py
```

---

## Suggested Repository Structure
```
.
├── action_recognition/
│   ├── train_actions.py
│   ├── action_model.h5
│   └── labels.json
├── emotion_recognition/
│   ├── train_emotions.py
│   ├── emotion_model.keras / .h5
│   └── labels.json
├── live_demo/
│   └── live_cam.py
└── README.md
```

---

## Skills Demonstrated
- Video preprocessing for deep learning (sequence building, sampling, padding)
- Transfer learning with pretrained CNN backbones (MobileNetV2)
- Temporal modeling using LSTM for action recognition
- Face alignment with Dlib landmarks and robust grayscale preprocessing (CLAHE + standardization)
- CNN architecture design using separable convolutions and residual blocks (Mini-Xception)
- Handling class imbalance (class weights, focal loss, mixup)
- Training optimization and comparison (SGD/Adam/Adagrad/AdamW + scheduling)
- Real-time deployment considerations (smoothing, thresholds, FPS/latency monitoring)
- Model export and label mapping for inference

---

## Educational Context
This implementation follows the Machine Vision major task requirements: a unified CNN–LSTM action model and a CNN emotion model integrated into a real-time webcam pipeline with live overlays and performance measurement.
