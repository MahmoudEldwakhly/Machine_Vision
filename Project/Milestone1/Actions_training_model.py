import os, zipfile, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, TimeDistributed,
    GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================================================
# CONFIG
# ======================================================
IMG_SIZE = 112    # frame size (H, W)
SEQ_LEN  = 20     # number of frames per video
BATCH    = 2      # videos per batch

# ======================================================
# AUGMENTATION (TRAIN ONLY)
# ======================================================
augmentor = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
)

# ======================================================
# VIDEO LOADER (WITH OPTIONAL AUGMENTATION)
# ======================================================
def load_video_fixed(path, seq_len, augment=False):
    """
    Load a video from 'path' and return exactly 'seq_len' frames,
    resized to (IMG_SIZE, IMG_SIZE), normalized to [0,1].
    If augment=True, apply random image augmentations.
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0

        if augment:
            frame = augmentor.random_transform(frame)

        frames.append(frame)

    cap.release()

    # Safety: if video couldn't be read at all
    if len(frames) == 0:
        frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)] * seq_len

    # If too short, pad by repeating last frame
    if len(frames) < seq_len:
        frames += [frames[-1]] * (seq_len - len(frames))

    # If too long, sample seq_len frames uniformly
    if len(frames) > seq_len:
        idxs = np.linspace(0, len(frames)-1, seq_len).astype(int)
        frames = [frames[i] for i in idxs]

    return np.array(frames, dtype=np.float32)


# ======================================================
# ZIP FILES
# ======================================================
zip_files = {
    "BodyWeightSquats": "/content/BodyWeightSquats.zip",
    "JumpingJack":      "/content/JumpingJack.zip",
    "PlayingGuitar":    "/content/PlayingGuitar.zip",
    "PlayingPiano":     "/content/PlayingPiano.zip",
}

extract_root = "/content/actions_data"
os.makedirs(extract_root, exist_ok=True)

# ======================================================
# FLATTENING ZIP EXTRACTION
# ======================================================
def extract_zip(label, zip_path):
    """
    Extract 'zip_path' into 'extract_root/label',
    flattening any directory structure inside the zip.
    """
    out_dir = os.path.join(extract_root, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nExtracting (flattening) {label}...")

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            # skip directories
            if member.endswith("/"):
                continue

            data = z.read(member)
            fname = os.path.basename(member)
            if not fname:
                continue

            target_path = os.path.join(out_dir, fname)
            with open(target_path, "wb") as f:
                f.write(data)

    return out_dir


folders = {label: extract_zip(label, path) for label, path in zip_files.items()}

print("\nExtracted folders:")
for k, v in folders.items():
    print("•", k, "→", v)


# ======================================================
# COLLECT VIDEO PATHS + LABELS
# ======================================================
video_paths = []
video_labels = []

label_names = list(folders.keys())
label_to_idx = {label: i for i, label in enumerate(label_names)}

print("\nLabel mapping:", label_to_idx)

for label, folder in folders.items():
    label_id = label_to_idx[label]

    # videos are directly inside folder (flattened)
    exts = ("*.avi", "*.mp4", "*.mov", "*.mkv")
    all_videos = []
    for ext in exts:
        all_videos.extend(glob(os.path.join(folder, ext)))  # non-recursive

    print(f"{label}: {len(all_videos)} videos found.")

    for vid in all_videos:
        video_paths.append(vid)
        video_labels.append(label_id)

video_labels = np.array(video_labels)
print("\nTotal videos:", len(video_paths))


# ======================================================
# TRAIN / VAL / TEST SPLIT (ON PATHS)
#   - 80% train
#   - 10% validation
#   - 10% test
# ======================================================
paths_train, paths_temp, y_train, y_temp = train_test_split(
    video_paths,
    video_labels,
    test_size=0.20,
    stratify=video_labels,
    shuffle=True,
    random_state=42
)

paths_val, paths_test, y_val, y_test = train_test_split(
    paths_temp,
    y_temp,
    test_size=0.50,        # half of 20% -> 10% val, 10% test
    stratify=y_temp,
    shuffle=True,
    random_state=42
)

print("Train videos:", len(paths_train))
print("Val videos:  ", len(paths_val))
print("Test videos: ", len(paths_test))


# Steps for generators
train_steps = max(1, len(paths_train) // BATCH)
val_steps   = max(1, len(paths_val)   // BATCH)
test_steps  = max(1, len(paths_test)  // BATCH)

print("\nSteps per epoch (train):", train_steps)
print("Validation steps:        ", val_steps)
print("Test steps:              ", test_steps)


# ======================================================
# VIDEO BATCH GENERATOR (NO FULL DATA IN RAM)
# ======================================================
def video_batch_generator(paths, labels, batch_size, augment=False):
    """
    Keras-style generator that yields (X_batch, y_batch) forever.
    X_batch shape: (batch_size, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)
    y_batch shape: (batch_size,)
    """
    paths = np.array(paths)
    labels = np.array(labels)

    while True:
        idxs = np.arange(len(paths))
        np.random.shuffle(idxs)

        for i in range(0, len(paths), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch_paths = paths[batch_idxs]
            batch_labels = labels[batch_idxs]

            batch_frames = []
            for p in batch_paths:
                frames = load_video_fixed(p, SEQ_LEN, augment=augment)
                batch_frames.append(frames)

            X_batch = np.array(batch_frames, dtype=np.float32)
            y_batch = np.array(batch_labels)

            yield X_batch, y_batch


# ======================================================
# MODEL BUILDER (MOBILENETV2 + LSTM, PARTIAL FINE-TUNING)
# ======================================================
def build_model():
    """
    Build the CNN+LSTM model:
      - MobileNetV2 backbone (pretrained on ImageNet)
      - TimeDistributed over frames
      - LSTM for temporal modeling
      - Dense softmax for classification
    """
    base_cnn = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze all CNN layers first
    base_cnn.trainable = False

    # fine-tune last 20 layers
    for layer in base_cnn.layers[-20:]:
        layer.trainable = True

    model = Sequential([
        TimeDistributed(base_cnn, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(128, return_sequences=False),
        Dense(len(label_names), activation="softmax")
    ])

    return model


# ======================================================
# TRAIN WITH MULTIPLE OPTIMIZERS
#   - Store metrics for train/val/test
#   - Store confusion matrices and sample test cases
# ======================================================
optimizers = {
    "SGD":     tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    "Adam":    tf.keras.optimizers.Adam(learning_rate=1e-4),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.001),
}

results             = {}  # summary metrics
histories           = {}  # training history (per epoch)
confusion_matrices  = {}  # confusion matrix per optimizer (on test set)
optimizer_test_cases = {} # sample test cases per optimizer

best_acc     = -1.0
best_name    = None
best_weights = None      # store weights instead of full model to avoid memory issues


for name, opt in optimizers.items():
    print("\n===================================")
    print(f"Training with optimizer: {name}")
    print("===================================")

    # Clear previous TF graph to reduce memory usage
    tf.keras.backend.clear_session()

    # Build a fresh model for this optimizer
    model = build_model()

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    # Generators for training and validation
    train_gen = video_batch_generator(paths_train, y_train, BATCH, augment=True)
    val_gen   = video_batch_generator(paths_val,   y_val,   BATCH, augment=False)

    # Train
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )
    duration = time.time() - start_time  # training time in seconds

    # Store history for curves
    histories[name] = history.history

    # Evaluate on train / val / test using fresh non-augmented generators
    train_eval_gen = video_batch_generator(paths_train, y_train, BATCH, augment=False)
    val_eval_gen   = video_batch_generator(paths_val,   y_val,   BATCH, augment=False)
    test_eval_gen  = video_batch_generator(paths_test,  y_test,  BATCH, augment=False)

    train_loss, train_acc = model.evaluate(train_eval_gen, steps=train_steps, verbose=0)
    val_loss,   val_acc   = model.evaluate(val_eval_gen,   steps=val_steps,   verbose=0)
    test_loss,  test_acc  = model.evaluate(test_eval_gen,  steps=test_steps,  verbose=0)

    print(f"\n{name} -> Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    print(f"{name} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"{name} -> Training Time: {duration:.1f} seconds (~{duration/60:.2f} minutes)")

    # Save metrics in 'results'
    results[name] = {
        "time_sec": float(duration),
        "time_min": float(duration / 60.0),
        "train_loss": float(train_loss),
        "train_acc":  float(train_acc),
        "val_loss":   float(val_loss),
        "val_acc":    float(val_acc),
        "test_loss":  float(test_loss),
        "test_acc":   float(test_acc),
    }

    # -----------------------------------------------------------------
    # Confusion matrix and sample test cases FOR THIS OPTIMIZER
    # -----------------------------------------------------------------
    y_true = []
    y_pred = []
    test_cases = []

    print(f"\nComputing predictions on TEST set for optimizer: {name} ...")
    for path, true_label in zip(paths_test, y_test):
        frames = load_video_fixed(path, SEQ_LEN, augment=False)
        probs = model.predict(frames[np.newaxis, ...], verbose=0)
        pred_label = int(np.argmax(probs, axis=1)[0])

        y_true.append(int(true_label))
        y_pred.append(pred_label)

        # Store info for testing cases (we'll print a few later)
        test_cases.append({
            "video_path": path,
            "true_label_idx": int(true_label),
            "true_label_name": label_names[int(true_label)],
            "pred_label_idx": pred_label,
            "pred_label_name": label_names[pred_label],
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    confusion_matrices[name] = cm
    optimizer_test_cases[name] = test_cases

    print(f"\nClassification report for optimizer: {name}")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # Track best optimizer based on TEST accuracy
    if test_acc > best_acc:
        best_acc     = test_acc
        best_name    = name
        best_weights = model.get_weights()  # save weights of best model

# ======================================================
# SUMMARY OF OPTIMIZERS
# ======================================================
print("\n============== SUMMARY (PER OPTIMIZER) ==============")
for name, info in results.items():
    print(f"{name}:")
    print(f"  Time:       {info['time_sec']:.1f} sec (~{info['time_min']:.2f} min)")
    print(f"  Train Acc:  {info['train_acc']:.4f},  Val Acc: {info['val_acc']:.4f},  Test Acc: {info['test_acc']:.4f}")
    print(f"  Train Loss: {info['train_loss']:.4f}, Val Loss: {info['val_loss']:.4f}, Test Loss: {info['test_loss']:.4f}")

print("\nBest optimizer:", best_name, "| Best Test Acc:", best_acc)


# ======================================================
# REBUILD BEST MODEL AND LOAD BEST WEIGHTS (FOR SAVING)
# ======================================================
tf.keras.backend.clear_session()
best_model = build_model()
if best_weights is not None:
    best_model.set_weights(best_weights)
else:
    print("WARNING: best_weights is None, something went wrong in selection.")


# ======================================================
# TRAINING CURVES (ACCURACY & LOSS VS EPOCHS)
# ======================================================
plt.figure(figsize=(14,5))

# Accuracy vs Epochs
plt.subplot(1,2,1)
for name, h in histories.items():
    plt.plot(h['accuracy'], label=f'{name} Train')
    plt.plot(h['val_accuracy'], linestyle='--', label=f'{name} Val')
plt.title("Accuracy vs Epochs (Train vs Val)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss vs Epochs
plt.subplot(1,2,2)
for name, h in histories.items():
    plt.plot(h['loss'], label=f'{name} Train')
    plt.plot(h['val_loss'], linestyle='--', label=f'{name} Val')
plt.title("Loss vs Epochs (Train vs Val)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ======================================================
# SUMMARY BAR PLOTS:
#   - Training time (seconds + minutes)
#   - Accuracy and Loss for train/val/test
# ======================================================
optimizer_names = list(results.keys())

times_sec = [results[n]['time_sec'] for n in optimizer_names]
train_accs = [results[n]['train_acc'] for n in optimizer_names]
val_accs   = [results[n]['val_acc']   for n in optimizer_names]
test_accs  = [results[n]['test_acc']  for n in optimizer_names]

train_losses = [results[n]['train_loss'] for n in optimizer_names]
val_losses   = [results[n]['val_loss']   for n in optimizer_names]
test_losses  = [results[n]['test_loss']  for n in optimizer_names]

# --- Training time bar plot ---
plt.figure(figsize=(8,5))
bars = plt.bar(optimizer_names, times_sec)
plt.title("Training Time per Optimizer (seconds)")
plt.ylabel("Time (sec)")
for bar, sec in zip(bars, times_sec):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0,
             height,
             f"{sec:.1f}s\n(~{sec/60:.2f}m)",
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# --- Accuracy bar plot (Train / Val / Test) ---
x = np.arange(len(optimizer_names))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, train_accs, width, label='Train Acc')
plt.bar(x,         val_accs,   width, label='Val Acc')
plt.bar(x + width, test_accs,  width, label='Test Acc')

plt.xticks(x, optimizer_names)
plt.ylabel("Accuracy")
plt.title("Accuracy (Train / Validation / Test) per Optimizer")
plt.legend()
plt.tight_layout()
plt.show()

# --- Loss bar plot (Train / Val / Test) ---
plt.figure(figsize=(10,6))
plt.bar(x - width, train_losses, width, label='Train Loss')
plt.bar(x,         val_losses,   width, label='Val Loss')
plt.bar(x + width, test_losses,  width, label='Test Loss')

plt.xticks(x, optimizer_names)
plt.ylabel("Loss")
plt.title("Loss (Train / Validation / Test) per Optimizer")
plt.legend()
plt.tight_layout()
plt.show()


# ======================================================
# CONFUSION MATRICES FOR ALL 3 OPTIMIZERS (ON TEST SET)
# ======================================================
num_opts = len(confusion_matrices)
plt.figure(figsize=(6 * num_opts, 5))

for i, (name, cm) in enumerate(confusion_matrices.items()):
    plt.subplot(1, num_opts, i + 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=label_names,
        yticklabels=label_names,
        cmap='Blues'
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({name}) - Test Set")

plt.tight_layout()
plt.show()


# ======================================================
# PRINT SAMPLE TEST CASES FOR EACH OPTIMIZER
# ======================================================
for name, cases in optimizer_test_cases.items():
    print("\n===================================")
    print(f"SAMPLE TEST CASES FOR OPTIMIZER: {name}")
    print("===================================")
    # print up to 5 examples
    for case in cases[:5]:
        print(f"Video: {os.path.basename(case['video_path'])}")
        print(f"  True : [{case['true_label_idx']}] {case['true_label_name']}")
        print(f"  Pred : [{case['pred_label_idx']}] {case['pred_label_name']}")
        print("-" * 40)


# ======================================================
# CONFUSION MATRIX FOR BEST OPTIMIZER (RE-PRINT, OPTIONAL)
# ======================================================
best_cm = confusion_matrices[best_name]

plt.figure(figsize=(6,5))
sns.heatmap(
    best_cm,
    annot=True,
    fmt='d',
    xticklabels=label_names,
    yticklabels=label_names,
    cmap='Blues'
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Best optimizer: {best_name}) - Test Set")
plt.tight_layout()
plt.show()


# ======================================================
# SAVE BEST MODEL + LABELS
# ======================================================
best_model.save("/content/action_model.h5")
with open("/content/labels.json", "w") as f:
    json.dump(label_names, f)

print("\nBest model saved to /content/action_model.h5")
print("Labels saved to /content/labels.json")
