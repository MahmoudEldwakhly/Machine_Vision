

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)


# ============================================================
# 1) LOAD CIFAR-10
# ============================================================
(x_train_full, y_train_full), (x_test_raw, y_test_raw) = tf.keras.datasets.cifar10.load_data()
y_train_full = y_train_full.squeeze()
y_test_raw   = y_test_raw.squeeze()

class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

print("=== DATASET OVERVIEW (CIFAR-10) ===")
print("Train (full):", x_train_full.shape, y_train_full.shape)
print("Test        :", x_test_raw.shape, y_test_raw.shape)
print("Pixel range (train full):", (x_train_full.min(), x_train_full.max()))
print("Classes:", class_names)

# ============================================================
# 2) DATASET EXPLORATION (distribution + samples)
# ============================================================
def plot_class_distribution(labels, title):
    counts = np.bincount(labels, minlength=10)
    plt.figure(figsize=(10,3))
    plt.bar(range(10), counts)
    plt.xticks(range(10), class_names, rotation=20, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return counts

def show_samples(images, labels, n=12, title="Random Samples"):
    idx = np.random.choice(len(images), n, replace=False)
    cols = 6
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 2.4*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[k])
        plt.title(class_names[labels[k]], fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

train_counts = plot_class_distribution(y_train_full, "Class Distribution - Train (Full)")
test_counts  = plot_class_distribution(y_test_raw,   "Class Distribution - Test")
print("Train class counts:", train_counts)
print("Test  class counts:", test_counts)

show_samples(x_train_full, y_train_full, n=12, title="Random CIFAR-10 Training Samples")

# ============================================================
# 3) EXPLICIT SPLIT: Train / Validation / Test
# ============================================================
val_ratio = 0.1
n = len(x_train_full)
perm = np.random.permutation(n)
val_size = int(n * val_ratio)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

x_train_raw, y_train = x_train_full[train_idx], y_train_full[train_idx]
x_val_raw,   y_val   = x_train_full[val_idx],   y_train_full[val_idx]
x_test,      y_test  = x_test_raw,              y_test_raw

print("\n=== SPLIT SIZES ===")
print("Train:", x_train_raw.shape, y_train.shape)
print("Val  :", x_val_raw.shape,   y_val.shape)
print("Test :", x_test.shape,      y_test.shape)

# ============================================================
# 4) PREPROCESS for pre-trained model
#    - We'll use MobileNetV2, include_top=False
#    - Resize CIFAR-10 from 32x32 -> 96x96 (lighter than 224)
#    - Use mobilenet_v2.preprocess_input
# ============================================================
IMG_SIZE = 96  # trade-off: faster than 224, compatible with MobileNetV2

def preprocess_mobilenet(x_uint8):
    x = tf.cast(x_uint8, tf.float32)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scales to [-1,1]
    return x

# Build tf.data pipelines (faster + cleaner)
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((x_train_raw, y_train)).shuffle(20000, seed=42).batch(BATCH_SIZE).map(
    lambda x, y: (preprocess_mobilenet(x), y), num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val_raw, y_val)).batch(BATCH_SIZE).map(
    lambda x, y: (preprocess_mobilenet(x), y), num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).map(
    lambda x, y: (preprocess_mobilenet(x), y), num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# ============================================================
# 5) LOAD PRE-TRAINED MODEL (include_top=False) + FREEZE BASE
# ============================================================
base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # freeze convolutional base

# Custom head for CIFAR-10
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== TRANSFER MODEL SUMMARY ===")
model.summary()
print("\nTrainable layers check:")
print("- Base (MobileNetV2) trainable?", base.trainable)
print("- Total trainable variables:", len(model.trainable_variables))

# ============================================================
# 6) TRAIN ONLY THE NEW LAYERS (5-10 epochs)
# ============================================================
EPOCHS = 8  # within hint range (5–10)
start_train = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
train_time_sec = time.time() - start_train

# ============================================================
# 7) PLOT TRAINING & VALIDATION CURVES (LOSS + ACCURACY)
# ============================================================
def plot_training_curves(hist):
    h = hist.history
    ep = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(ep, h["loss"], label="Train Loss")
    plt.plot(ep, h["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curves (Transfer Learning)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, h["accuracy"], label="Train Accuracy")
    plt.plot(ep, h["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curves (Transfer Learning)")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)

# ============================================================
# 8) EVALUATE ON TEST SET
# ============================================================
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print("\n=== TEST PERFORMANCE (Transfer Learning) ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {train_time_sec:.2f} seconds for {EPOCHS} epochs")

# ============================================================
# 9) CONFUSION MATRIX (COLORED) ON TEST
# ============================================================
# Predict on test in one go for CM + samples
all_probs = model.predict(test_ds, verbose=0)

# Because test_ds is batched, the prediction order matches y_test order,
# as long as test_ds is created in the same order (it is).
y_pred = np.argmax(all_probs, axis=1)

def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = confusion_matrix_np(y_test, y_pred, 10)

plt.figure(figsize=(9,7))
plt.imshow(cm, cmap="plasma")
plt.title("Confusion Matrix (CIFAR-10 Test) - Transfer Learning (Colored)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(10), class_names, rotation=25, ha="right")
plt.yticks(range(10), class_names)
plt.colorbar()

thresh = cm.max() * 0.6
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)
plt.tight_layout()
plt.show()

# Most common confusions
off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
pairs = []
for i in range(10):
    for j in range(10):
        if i != j and off_diag[i, j] > 0:
            pairs.append((off_diag[i, j], i, j))
pairs.sort(reverse=True)

print("\n=== MOST COMMON CONFUSIONS (true -> predicted) ===")
for c, t, p in pairs[:10]:
    print(f"- {class_names[t]} -> {class_names[p]}: {c} times")

# ============================================================
# 10) DISPLAY SAMPLE PREDICTIONS WITH TRUE LABELS (ON ORIGINAL 32x32 images)
# ============================================================
def show_sample_predictions(x_test_images_uint8, y_true, y_pred, probs, n=12):
    idx = np.random.choice(len(x_test_images_uint8), n, replace=False)
    cols = 6
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 2.4*rows))

    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_test_images_uint8[k])  # show original
        conf = float(np.max(probs[k]))
        correct = (y_pred[k] == y_true[k])
        mark = "✓" if correct else "✗"
        plt.title(f"T={class_names[y_true[k]]}\nP={class_names[y_pred[k]]}\n{conf*100:.1f}% {mark}", fontsize=9)
        plt.axis("off")

    plt.suptitle("Sample Predictions (CIFAR-10 Test) - Transfer Learning")
    plt.tight_layout()
    plt.show()

    print("\nSample Predictions (index | true | pred | confidence):")
    for k in idx:
        conf = float(np.max(probs[k]))
        print(f"- {k:5d} | {class_names[y_true[k]]:10s} | {class_names[y_pred[k]]:10s} | {conf:.4f}")

show_sample_predictions(x_test_raw, y_test, y_pred, all_probs, n=12)

# ============================================================
# 11) SMALL DETAILED REPORT
# ============================================================
train_loss_last = history.history["loss"][-1]
val_loss_last   = history.history["val_loss"][-1]
train_acc_last  = history.history["accuracy"][-1]
val_acc_last    = history.history["val_accuracy"][-1]

print("\n" + "="*60)
print("SMALL DETAILED REPORT (Transfer Learning on CIFAR-10)")
print("="*60)
print("What we did:")
print("1) Loaded CIFAR-10 and explored it (class distribution + sample images).")
print("2) Split into Train/Validation/Test.")
print("3) Loaded MobileNetV2 pre-trained on ImageNet with include_top=False.")
print("4) Froze the convolutional base and added a small custom classifier head for 10 classes.")
print("5) Trained only the new layers for a small number of epochs and evaluated on test.")
print("\nPreprocessing:")
print(f"- Resized images 32x32 -> {IMG_SIZE}x{IMG_SIZE}")
print("- Used MobileNetV2 preprocess_input (scales inputs to the format expected by the model).")
print("\nTraining:")
print(f"- Epochs: {EPOCHS}")
print(f"- Final Train Loss: {train_loss_last:.4f} | Final Val Loss: {val_loss_last:.4f}")
print(f"- Final Train Acc : {train_acc_last:.4f} | Final Val Acc : {val_acc_last:.4f}")
print(f"- Training time   : {train_time_sec:.2f} sec")
print("\nTesting + Analysis:")
print(f"- Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("- Plotted loss/accuracy curves to see learning progress.")
print("- Built a colored confusion matrix to see which classes are mixed up.")
print("- Displayed sample predictions with true labels + confidence.")
print("\nNotes:")
print("- Transfer learning often boosts accuracy because the pre-trained model already learned useful visual features.")
print("- If you want better results, you can: (a) train a few more epochs, or (b) unfreeze the last layers of the base and fine-tune with a very small learning rate.")
print("="*60)
