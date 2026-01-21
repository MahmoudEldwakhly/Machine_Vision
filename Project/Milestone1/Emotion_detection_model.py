"""
FER-2013 Emotion Recognition
Dlib 68-Landmark Face Alignment + Mini-Xception CNN


# ===========================================
# 1. IMPORTS
# ===========================================
import os
import math
import json
import urllib.request
import subprocess
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization, ReLU, Add,
    GlobalAveragePooling2D, Dropout, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ===========================================
# 2. PATHS & SETTINGS
# ===========================================
train_dir = "/content/datasets/train"
test_dir  = "/content/datasets/test"

img_size = (48, 48)
num_classes = 7
batch_size = 64
epochs = 50
val_size = 0.12

label_smoothing = 0.05
weight_decay_l2 = 1e-4

# Your chosen values
MIXUP_ALPHA = 0.25
FOCAL_GAMMA = 1.7

# Class-weight cap (requested)
CLASS_WEIGHT_CAP = 2.5

# Deploy artifacts (requested)
DEPLOY_MODEL_PATH = "/content/sample_data/action_model.keras"
LABELS_PATH = "/content/sample_data/labels.json"

# ===========================================
# 3. CHECK CLASSES
# ===========================================
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"train_dir not found: {train_dir}")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"test_dir not found: {test_dir}")

classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print("Classes:", classes)
if len(classes) != num_classes:
    print(f"WARNING: num_classes={num_classes}, but found {len(classes)} folders.")

# ===========================================
# 4. DLIB SETUP (download shape predictor if missing)
# ===========================================
PRED_PATH = "shape_predictor_68_face_landmarks.dat"
PRED_BZ2  = "shape_predictor_68_face_landmarks.dat.bz2"
PRED_URL  = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

def ensure_dlib_predictor():
    if os.path.exists(PRED_PATH):
        return
    print("Downloading Dlib shape predictor...")
    if not os.path.exists(PRED_BZ2):
        urllib.request.urlretrieve(PRED_URL, PRED_BZ2)
    print("Decompressing predictor (bz2)...")
    try:
        subprocess.run(["bzip2", "-dk", PRED_BZ2], check=True)
    except Exception as e:
        raise RuntimeError("Failed to decompress .bz2. Install bzip2 or decompress manually.") from e

ensure_dlib_predictor()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)

# ===========================================
# 5. PREPROCESSING: ALIGNMENT + CLAHE + STANDARDIZATION
# ===========================================
def preprocess_gray(gray48: np.ndarray) -> np.ndarray:
    """CLAHE + per-image standardization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray48)
    g = g.astype(np.float32)
    mean, std = float(g.mean()), float(g.std())
    g = (g - mean) / (std + 1e-6)
    return g

def align_face_dlib(img, output_size=(48,48), expand_ratio=0.30):
    """
    Face alignment using:
      - Dlib HOG detector
      - 68 landmarks
      - Eye-alignment rotation + expanded crop
    Returns: float32 (48,48) after preprocess_gray
    """
    if img is None:
        return np.zeros(output_size, np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        g = cv2.resize(gray, output_size)
        return preprocess_gray(g)

    # choose the largest face (fixes rects[0] issue)
    rect = max(rects, key=lambda r: r.width() * r.height())

    shape = predictor(gray, rect)
    shape_np = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

    left_eye  = shape_np[36:42].mean(axis=0).astype(np.int32)
    right_eye = shape_np[42:48].mean(axis=0).astype(np.int32)

    dy = int(right_eye[1] - left_eye[1])
    dx = int(right_eye[0] - left_eye[0])
    angle = float(np.degrees(np.arctan2(dy, dx)))

    eye_center = tuple(((left_eye + right_eye) // 2).tolist())
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, rot_mat, (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    x = max(0, int(x - w * expand_ratio))
    y = max(0, int(y - h * expand_ratio))
    w = int(w * (1 + 2 * expand_ratio))
    h = int(h * (1 + 2 * expand_ratio))

    x2 = min(rotated.shape[1], x + w)
    y2 = min(rotated.shape[0], y + h)

    crop = rotated[y:y2, x:x2]
    if crop.size == 0:
        g = cv2.resize(gray, output_size)
        return preprocess_gray(g)

    aligned = cv2.resize(crop, output_size)
    return preprocess_gray(aligned)

# ===========================================
# 6. LOAD DATASET
# ===========================================
def load_dataset(folder):
    X, y = [], []
    class_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        folder_cls = os.path.join(folder, cls)
        if not os.path.isdir(folder_cls):
            continue

        for file in os.listdir(folder_cls):
            img_path = os.path.join(folder_cls, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            aligned = align_face_dlib(img, output_size=img_size)
            X.append(aligned)
            y.append(class_map[cls])

    X = np.array(X, dtype=np.float32).reshape(-1, 48, 48, 1)
    y = tf.keras.utils.to_categorical(y, num_classes)
    print(f"Loaded {X.shape[0]} images from {folder}")
    return X, y

print("\nLoading full training set...")
X_all, y_all = load_dataset(train_dir)

print("Loading test set (kept untouched)...")
X_test, y_test = load_dataset(test_dir)

# Train/Val split (stratified)
y_all_labels = np.argmax(y_all, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=val_size,
    random_state=42,
    stratify=y_all_labels
)
print(f"\nSplit: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

# Class weights
y_train_labels = np.argmax(y_train, axis=1)
cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train_labels)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("Class weights (raw):", class_weights)

# Requested edit: cap class weights
class_weights = {k: float(min(v, CLASS_WEIGHT_CAP)) for k, v in class_weights.items()}
print(f"Class weights (CAPPED at {CLASS_WEIGHT_CAP}):", class_weights)

# ===========================================
# 7. DATA AUGMENTATION
# ===========================================
datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=(0.85, 1.15)
)
datagen.fit(X_train)

# ===========================================
# 8. MIXUP WRAPPER
# ===========================================
def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(x.shape[0])
    x2, y2 = x[idx], y[idx]
    return lam * x + (1 - lam) * x2, lam * y + (1 - lam) * y2

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_gen, alpha=0.2):
        self.base_gen = base_gen
        self.alpha = alpha
    def __len__(self):
        return len(self.base_gen)
    def __getitem__(self, i):
        x, y = self.base_gen[i]
        return mixup_batch(x, y, self.alpha)

# ===========================================
# 9. MINI-XCEPTION
# ===========================================
def mini_xception(input_shape=(48,48,1), n_classes=7, wd=1e-4):
    inp = Input(input_shape)

    x = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(wd))(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, (3,3), strides=2, padding='same', kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    def res_block(x, filters):
        res = Conv2D(filters, (1,1), strides=2, padding='same', kernel_regularizer=l2(wd))(x)

        y = SeparableConv2D(filters, (3,3), padding='same',
                            depthwise_regularizer=l2(wd), pointwise_regularizer=l2(wd))(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)

        y = SeparableConv2D(filters, (3,3), strides=2, padding='same',
                            depthwise_regularizer=l2(wd), pointwise_regularizer=l2(wd))(y)
        y = BatchNormalization()(y)

        out = Add()([res, y])
        out = ReLU()(out)
        return out

    x = res_block(x, 128)
    x = res_block(x, 256)
    x = res_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation="softmax", kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

# ===========================================
# 10. LOSS: Mild Focal CE with label smoothing
# ===========================================
def focal_categorical_loss(gamma=1.7, label_smoothing=0.0):
    ce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing, reduction="none")
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        base = ce(y_true, y_pred)  # per-sample
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)  # prob of true class
        focal = tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(focal * base)
    return loss

loss_fn = focal_categorical_loss(gamma=FOCAL_GAMMA, label_smoothing=label_smoothing)

# ===========================================
# 11. OPTIMIZERS TO COMPARE (added AdamW)
# ===========================================
def make_optimizer(name: str, steps_per_epoch: int = None, epochs: int = None):
    name = name.lower().strip()

    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.0, nesterov=False)

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=3e-4)

    if name == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=0.01)

    if name == "adamw":
        if steps_per_epoch is None or epochs is None:
            raise ValueError("adamw requires steps_per_epoch and epochs for cosine decay schedule.")
        total_steps = steps_per_epoch * epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=8e-4,
            decay_steps=total_steps,
            alpha=1e-2
        )
        return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    raise ValueError(f"Unknown optimizer: {name}")

# ===========================================
# 12. TRAIN ONE RUN (per optimizer)
# ===========================================
def train_with_optimizer(opt_name: str, steps_per_epoch: int):
    tf.keras.backend.clear_session()

    model = mini_xception(input_shape=(48,48,1), n_classes=num_classes, wd=weight_decay_l2)
    optimizer = make_optimizer(opt_name, steps_per_epoch=steps_per_epoch, epochs=epochs)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    ckpt_path = f"best_{opt_name}.h5"
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True, verbose=1),
    ]

    base_gen = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    train_gen = MixupGenerator(base_gen, alpha=MIXUP_ALPHA)

    print(f"\n==============================")
    print(f"Training with optimizer: {opt_name}")
    print(f"==============================")

    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    best_val_acc = float(np.max(history.history["val_accuracy"]))
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

    return model, history, best_val_acc, best_epoch, ckpt_path

# ===========================================
# 13. RUN COMPARISON
# ===========================================
steps_per_epoch = math.ceil(len(X_train) / batch_size)

optimizers_to_test = ["sgd", "adam", "adagrad", "adamw"]
results = []
histories = {}

for opt in optimizers_to_test:
    model, history, best_val_acc, best_epoch, ckpt_path = train_with_optimizer(opt, steps_per_epoch=steps_per_epoch)
    results.append((opt, best_val_acc, best_epoch, ckpt_path))
    histories[opt] = history

# Print comparison table
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print("\n\n=== Optimizer Comparison (by best Val Accuracy) ===")
print(f"{'Optimizer':<10} | {'Best Val Acc':<12} | {'Best Epoch':<10} | {'Checkpoint'}")
print("-"*70)
for opt, acc, ep, ck in results_sorted:
    print(f"{opt:<10} | {acc:<12.4f} | {ep:<10d} | {ck}")

best_opt, best_acc, best_ep, best_ckpt = results_sorted[0]
print(f"\nBEST OPTIMIZER: {best_opt}  (best val acc={best_acc:.4f} at epoch {best_ep})")
print(f"Best checkpoint saved at: {best_ckpt}")

# Load best weights into a fresh model (safer)
best_model = mini_xception(input_shape=(48,48,1), n_classes=num_classes, wd=weight_decay_l2)
best_model.compile(optimizer=make_optimizer(best_opt, steps_per_epoch=steps_per_epoch, epochs=epochs),
                   loss=loss_fn, metrics=["accuracy"])
best_model.load_weights(best_ckpt)

# ===========================================
# 14. EVALUATE ON TEST (HONEST) USING BEST OPT
# ===========================================
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTEST accuracy (best optimizer={best_opt}): {test_acc:.4f}")

y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report (TEST):\n")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (TEST) - Best Optimizer: {best_opt}")
plt.show()

# ===========================================
# 15. PLOT 1: Validation Accuracy Comparison
# ===========================================
plt.figure(figsize=(12,5))
for opt in optimizers_to_test:
    plt.plot(histories[opt].history["val_accuracy"], label=f"{opt} val_acc")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# 16. PLOT 2: For each optimizer -> Accuracy + Loss (train/val)
# ===========================================
for opt in optimizers_to_test:
    h = histories[opt].history

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(h["accuracy"], label="Train")
    plt.plot(h["val_accuracy"], label="Val")
    plt.title(f"Accuracy - {opt}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(h["loss"], label="Train")
    plt.plot(h["val_loss"], label="Val")
    plt.title(f"Loss - {opt}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ===========================================
# 17. SAVE FINAL BEST MODEL + labels.json (requested)
# ===========================================
best_model.save(DEPLOY_MODEL_PATH)
print(f"\nSaved deployable model to: {DEPLOY_MODEL_PATH}")

labels = {str(i): cls for i, cls in enumerate(classes)}
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)
print(f"Saved labels mapping to: {LABELS_PATH}")
print("Labels:", labels)

print("\nDone.")
print(f"Best checkpoint: {best_ckpt}")
print(f"Saved deployable model: {DEPLOY_MODEL_PATH}")
print(f"Saved labels file: {LABELS_PATH}")