
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# ============================================================
# 1) LOAD DATA
# ============================================================
(x_train_full, y_train_full), (x_test_raw, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("=== DATASET OVERVIEW (Fashion-MNIST) ===")
print("Train (full):", x_train_full.shape, y_train_full.shape)
print("Test        :", x_test_raw.shape, y_test.shape)
print("Pixel range (train full):", (x_train_full.min(), x_train_full.max()))
print("Pixel range (test):      ", (x_test_raw.min(), x_test_raw.max()))
print("Classes:", class_names)

# ============================================================
# 2) DATASET EXPLORATION (START OF CODE)
# ============================================================
def plot_class_distribution(labels, title):
    counts = np.bincount(labels, minlength=10)
    plt.figure(figsize=(10,3))
    plt.bar(range(10), counts)
    plt.xticks(range(10), class_names, rotation=35, ha="right")
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
    plt.figure(figsize=(12, 2.2*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[k], cmap="gray")
        plt.title(class_names[labels[k]], fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Class distribution (train full + test)
train_counts = plot_class_distribution(y_train_full, "Class Distribution - Train (Full)")
test_counts  = plot_class_distribution(y_test,       "Class Distribution - Test")

print("Train class counts:", train_counts)
print("Test  class counts:", test_counts)

# Show sample images from training set
show_samples(x_train_full, y_train_full, n=12, title="Random Samples from Training Set")

# ============================================================
# 3) EXPLICIT SPLIT: Train / Validation / Test
#    - We'll split the original training set into train+val.
# ============================================================
val_ratio = 0.1  # 10% validation from training set
num_train = len(x_train_full)
idx = np.random.permutation(num_train)

val_size = int(num_train * val_ratio)
val_idx = idx[:val_size]
train_idx = idx[val_size:]

x_val_raw, y_val = x_train_full[val_idx], y_train_full[val_idx]
x_train_raw, y_train = x_train_full[train_idx], y_train_full[train_idx]

print("\n=== SPLIT SIZES ===")
print("Train:", x_train_raw.shape, y_train.shape)
print("Val  :", x_val_raw.shape,   y_val.shape)
print("Test :", x_test_raw.shape,  y_test.shape)

# ============================================================
# 4) PREPROCESS: normalize + add channel dimension
# ============================================================
def preprocess_for_cnn(x):
    x = x.astype("float32") / 255.0
    x = x[..., np.newaxis]   # (N, 28, 28, 1)
    return x

x_train = preprocess_for_cnn(x_train_raw)
x_val   = preprocess_for_cnn(x_val_raw)
x_test  = preprocess_for_cnn(x_test_raw)

# ============================================================
# 5) BUILD A SHALLOW CNN (Conv -> MaxPool) + Dense
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== MODEL SUMMARY ===")
model.summary()

# ============================================================
# 6) TRAIN (10 epochs) using explicit validation data
# ============================================================
EPOCHS = 10
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=128,
    verbose=1
)

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
    plt.title("Loss Curves (CNN)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, h["accuracy"], label="Train Accuracy")
    plt.plot(ep, h["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curves (CNN)")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)

# ============================================================
# 8) EVALUATE ON TEST SET
# ============================================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n=== TEST PERFORMANCE ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ============================================================
# 9) CONFUSION MATRIX (COLORED) ON TEST
# ============================================================
all_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(all_probs, axis=1)

def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = confusion_matrix_np(y_test, y_pred, num_classes=10)

plt.figure(figsize=(9,7))
plt.imshow(cm, cmap="viridis")
plt.title("Confusion Matrix (Fashion-MNIST CNN Test) - Colored")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(range(10), class_names, rotation=35, ha="right")
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
# 10) DISPLAY SAMPLE PREDICTIONS (TRUE vs PREDICTED)
# ============================================================
def show_sample_predictions(x_images, y_true, probs, n=12):
    idx = np.random.choice(len(x_images), n, replace=False)
    preds = np.argmax(probs[idx], axis=1)
    confs = np.max(probs[idx], axis=1)

    cols = 6
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 2.2*rows))

    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_images[k].squeeze(), cmap="gray")
        correct = preds[i] == y_true[k]
        mark = "✓" if correct else "✗"
        plt.title(
            f"T={class_names[y_true[k]]}\nP={class_names[preds[i]]}\n{confs[i]*100:.1f}% {mark}",
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle("Sample Predictions on TEST Set (True vs Predicted)")
    plt.tight_layout()
    plt.show()

    print("\nSample Predictions (index | true | pred | confidence):")
    for i, k in enumerate(idx):
        print(f"- {k:5d} | {class_names[y_true[k]]:12s} | {class_names[preds[i]]:12s} | {confs[i]:.4f}")

show_sample_predictions(x_test, y_test, all_probs, n=12)

# ============================================================
# 11) SMALL DETAILED REPORT
# ============================================================
train_loss_last = history.history["loss"][-1]
val_loss_last   = history.history["val_loss"][-1]
train_acc_last  = history.history["accuracy"][-1]
val_acc_last    = history.history["val_accuracy"][-1]

print("\n" + "="*60)
print("SMALL DETAILED REPORT (CNN on Fashion-MNIST)")
print("="*60)
print("Dataset:")
print("- Fashion-MNIST: 28x28 grayscale clothing images, 10 classes.")
print("- We explored the dataset with class distributions and sample images.")
print("\nData split:")
print(f"- Train: {len(x_train)} images | Validation: {len(x_val)} images | Test: {len(x_test)} images")
print("\nPreprocessing:")
print("- Normalized pixels to [0,1]. Added channel dimension => (28,28,1).")
print("\nModel:")
print("- Shallow CNN: Conv2D + MaxPool + Conv2D + MaxPool + Flatten + Dense + Dropout + Softmax.")
print("\nTraining (10 epochs):")
print(f"- Final Train Loss: {train_loss_last:.4f} | Final Val Loss: {val_loss_last:.4f}")
print(f"- Final Train Acc : {train_acc_last:.4f} | Final Val Acc : {val_acc_last:.4f}")
print("\nTesting:")
print(f"- Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("\nEvaluation:")
print("- Plotted loss/accuracy curves to monitor learning and overfitting.")
print("- Confusion matrix shows which clothing classes are most confused.")
print("- Sample predictions show real model outputs with confidence.")
print("="*60)
