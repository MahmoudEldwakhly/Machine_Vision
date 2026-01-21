# CIFAR-10 Dense-only NN + Dataset Exploration + Colored Confusion Matrix + Test Cases (Single Colab Cell)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------
# 1) Load CIFAR-10
# ----------------------------
(x_train_img, y_train), (x_test_img, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.squeeze()  # (50000,1) -> (50000,)
y_test  = y_test.squeeze()   # (10000,1) -> (10000,)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("=== DATASET OVERVIEW (CIFAR-10) ===")
print("Train images shape:", x_train_img.shape, " Train labels shape:", y_train.shape)
print("Test  images shape:", x_test_img.shape,  " Test  labels shape:", y_test.shape)
print("Pixel range (train):", (x_train_img.min(), x_train_img.max()))
print("Pixel range (test): ", (x_test_img.min(), x_test_img.max()))
print("Classes (0-9):", list(range(10)))
print("Class names:", class_names)

# ----------------------------
# 2) Explore dataset: distribution + sample images
# ----------------------------
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

train_counts = plot_class_distribution(y_train, "Train Class Distribution (CIFAR-10)")
test_counts  = plot_class_distribution(y_test,  "Test Class Distribution (CIFAR-10)")
print("Train class counts:", train_counts)
print("Test  class counts:", test_counts)

def show_samples(images, labels, n=12, title="Random Samples"):
    idx = np.random.choice(len(images), n, replace=False)
    cols = 6
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 2.4*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[k])
        plt.title(class_names[labels[k]])
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_samples(x_train_img, y_train, n=12, title="Random Training Samples (CIFAR-10)")

# ----------------------------
# 3) Preprocess: normalize + flatten
# ----------------------------
x_train = (x_train_img.astype("float32") / 255.0).reshape(-1, 32*32*3)
x_test  = (x_test_img.astype("float32")  / 255.0).reshape(-1, 32*32*3)

# ----------------------------
# 4) Build Dense-only model
#    (CIFAR-10 is harder for dense-only; keep it small-ish but capable)
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32*32*3,)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation="relu"),
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

# ----------------------------
# 5) Train (10 epochs)
# ----------------------------
EPOCHS = 10
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ----------------------------
# 6) Plot training + validation accuracy curves (Task requirement)
# ----------------------------
def plot_accuracy_curves(hist):
    h = hist.history
    ep = range(1, len(h["accuracy"]) + 1)
    plt.figure(figsize=(7,4))
    plt.plot(ep, h["accuracy"], label="Train Accuracy")
    plt.plot(ep, h["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves (CIFAR-10, Dense-only)")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_accuracy_curves(history)

# (Optional bonus: loss curves)
def plot_loss_curves(hist):
    h = hist.history
    ep = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(7,4))
    plt.plot(ep, h["loss"], label="Train Loss")
    plt.plot(ep, h["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (CIFAR-10, Dense-only)")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_loss_curves(history)

# ----------------------------
# 7) Test evaluation
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n=== TEST PERFORMANCE ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ----------------------------
# 8) Colored Confusion Matrix + Most common confusions
# ----------------------------
all_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(all_probs, axis=1)

def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

cm = confusion_matrix_np(y_test, y_pred, num_classes=10)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="plasma")   # colored heatmap
plt.title("Confusion Matrix (CIFAR-10 Test) - Colored")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(range(10), class_names, rotation=25, ha="right")
plt.yticks(range(10), class_names)
plt.colorbar()

# write counts on cells
thresh = cm.max() * 0.6
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)

plt.tight_layout()
plt.show()

# show most frequent confusions
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

# ----------------------------
# 9) Test cases (random predictions shown as images)
# ----------------------------
def predict_test_cases(x_test_images, y_true, y_pred, probs, n_cases=12):
    idx = np.random.choice(len(x_test_images), n_cases, replace=False)
    cols = 6
    rows = int(np.ceil(n_cases / cols))
    plt.figure(figsize=(12, 2.4*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_test_images[k])
        conf = np.max(probs[k])
        correct = (y_pred[k] == y_true[k])
        mark = "✓" if correct else "✗"
        plt.title(f"T={class_names[y_true[k]]}\nP={class_names[y_pred[k]]}\n{conf*100:.1f}% {mark}")
        plt.axis("off")
    plt.suptitle("Random Test Cases: True vs Predicted + Confidence (CIFAR-10)")
    plt.tight_layout()
    plt.show()

    print("\nSample Predictions (index, true, pred, confidence):")
    for k in idx:
        print(f"- idx={k:5d} | true={class_names[y_true[k]]:10s} | pred={class_names[y_pred[k]]:10s} | conf={np.max(probs[k]):.4f}")

predict_test_cases(x_test_img, y_test, y_pred, all_probs, n_cases=12)

# ----------------------------
# 10) Misclassified examples (visual)
# ----------------------------
def show_misclassified(x_test_images, y_true, y_pred, probs, n=12):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassifications found (rare).")
        return
    pick = np.random.choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)

    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    plt.figure(figsize=(12, 2.4*rows))
    for i, k in enumerate(pick):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_test_images[k])
        conf = np.max(probs[k])
        plt.title(f"T={class_names[y_true[k]]}\nP={class_names[y_pred[k]]}\n({conf*100:.1f}%)")
        plt.axis("off")
    plt.suptitle("Misclassified Test Examples (CIFAR-10)")
    plt.tight_layout()
    plt.show()

show_misclassified(x_test_img, y_test, y_pred, all_probs, n=12)

# ----------------------------
# 11) Small detailed report
# ----------------------------
train_acc_last = history.history["accuracy"][-1]
val_acc_last   = history.history["val_accuracy"][-1]
train_loss_last = history.history["loss"][-1]
val_loss_last   = history.history["val_loss"][-1]

print("\n" + "="*60)
print("SMALL DETAILED REPORT (Task 3: CIFAR-10)")
print("="*60)
print("Exploration:")
print("- Loaded CIFAR-10: 50,000 training images and 10,000 test images.")
print("- Each image is 32x32 with 3 color channels (RGB).")
print("- Displayed random sample images and plotted class distributions.")
print("\nPreprocessing:")
print("- Normalized pixel values to [0, 1].")
print("- Flattened each image from 32x32x3 to 3072 features.")
print("\nModel (Dense-only):")
print("- Dense(512, ReLU) + Dropout(0.4) + Dense(256, ReLU) + Dropout(0.3) + Dense(10, Softmax).")
print("\nTraining (10 epochs):")
print(f"- Final Train Acc: {train_acc_last:.4f} | Final Val Acc: {val_acc_last:.4f}")
print(f"- Final Train Loss: {train_loss_last:.4f} | Final Val Loss: {val_loss_last:.4f}")
print("\nTesting:")
print(f"- Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("\nNotes:")
print("- CIFAR-10 is harder than MNIST/Fashion-MNIST; Dense-only models usually perform worse than CNNs.")
print("- Confusion matrix and misclassified examples help identify which classes are most frequently mixed up.")
print("="*60)
