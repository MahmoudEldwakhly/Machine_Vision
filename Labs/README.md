# CSE480 — Machine Vision Labs (Fall 2025)

This folder collects hands-on lab exercises for **CSE480: Machine Vision**. The work covers classical image processing, feature-based tricks, and deep-learning baselines for vision tasks. Each lab is written as a small, focused pipeline with clear inputs/outputs and visual results.

---

## What’s inside

- **Lab 01 — Basic OpenCV image manipulation**
  - Create a fan-style **ID card**: load an image, resize to a fixed photo box, add text overlays, add a border, then save the final card.
  - Create a **social media frame**: flip the image horizontally, draw a colored frame (rounded look simulated with thick lines), and save the result.
  - Create a **transition scene**: resize two images to the same size and generate a cross-fade sequence using weighted blending, then save a final combined strip.

- **Lab 02 — Portrait mode + pencil sketch**
  - **Selective focus (portrait mode effect)**: build an elliptical mask for the foreground subject, blur the full image heavily, then merge sharp subject + blurred background using bitwise operations.
  - **Pencil sketch**:
    - Color-dodge style sketch using invert → blur → divide blending.
    - Edge-based sketch using Canny for sharper technical edges and comparison between methods.

- **Lab 03 — Artistic sketch + face region without pretrained detectors**
  - Enhance contrast with **histogram equalization**, then extract edges using **Laplacian** and **Canny** to compare visual differences.
  - Create a soft sketch look by blurring edges and blending them with the original image.
  - Detect an approximate face region using **skin segmentation in HSV + morphology + largest contour**, then keep the detected region sharp while blurring the background.

- **Lab 04 — Dense neural networks for image classification**
  - Dense-only networks for:
    - **MNIST**
    - **Fashion-MNIST**
    - **CIFAR-10**
  - Common pipeline used across tasks:
    - Dataset exploration (samples + class distribution)
    - Preprocessing (normalization + flattening)
    - Training curves (loss/accuracy)
    - Error analysis (confusion matrix + misclassified samples)

- **Lab 05 — CNN classification + transfer learning**
  - Train a **shallow CNN** on Fashion-MNIST and visualize learning curves and sample predictions.
  - Use a **pretrained backbone** (VGG16 / ResNet50 / MobileNet without top) on CIFAR-10:
    - Freeze convolutional base, add custom classifier head, train new layers, and compare behavior versus dense-only baselines.

- **Lab 06 — Classical ML baseline on Digits**
  - Load the **digits** dataset, visualize sample images, split train/test, standardize features, then train a **k-NN classifier** and evaluate on the test split.

---

## Suggested folder structure

```text
Machine-Vision/
  labs/
    lab01/
    lab02/
    lab03/
    lab04/
    lab05/
    lab06/
```

---

## Tech stack

- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## How to run

1. Create a virtual environment and install dependencies.
2. Run each lab script or notebook after updating input file paths (images/datasets).
3. Outputs are saved as images (and models for deep-learning labs when applicable).

---

## Learning outcomes

- Image I/O, resizing, geometric transforms, blending, masking, and bitwise composition.
- Histogram equalization, edge detection (Laplacian/Canny), and artistic post-processing.
- Region extraction using classical segmentation + morphology.
- Building baseline deep-learning classifiers and analyzing performance using curves and confusion matrices.
- Transfer learning workflow with frozen pretrained backbones.
- Classical ML pipeline: standardization + k-NN classification.
