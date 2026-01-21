# Fashion-MNIST
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------
# 1) Load Fashion-MNIST
# ----------------------------
(x_train_img, y_train), (x_test_img, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("=== DATASET OVERVIEW (Fashion-MNIST) ===")
print("Train images shape:", x_train_img.shape, " Train labels shape:", y_train.shape)
print("Test  images shape:", x_test_img.shape,  " Test  labels shape:", y_test.shape)
print("Pixel range (train):", (x_train_img.min(), x_train_img.max()))
print("Pixel range (test): ", (x_test_img.min(), x_test_img.max()))
print("Classes (0-9):", list(range(10)))
print("Class names:", class_names)

# ----------------------------
# 2) Explore dataset (distribution + sample images)
# ----------------------------
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

train_counts = plot_class_distribution(y_train, "Train Class Distribution (Fashion-MNIST)")
test_counts  = plot_class_distribution(y_test,  "Test Class Distribution (Fashion-MNIST)")

print("Train class counts:", train_counts)
print("Test  class counts:", test_counts)

def show_samples(images, labels, n=12, title="Random Samples"):
    idx = np.random.choice(len(images), n, replace=False)
    cols = 6
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 2.2*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[k], cmap="gray")
        plt.title(class_names[labels[k]])
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_samples(x_train_img, y_train, n=12, title="Random Training Samples (Fashion-MNIST)")

# ----------------------------
# 3) Normalize + Flatten
# ----------------------------
x_train = (x_train_img.astype("float32") / 255.0).reshape(-1, 28*28)
x_test  = (x_test_img.astype("float32")  / 255.0).reshape(-1, 28*28)

# ----------------------------
# 4) Build Dense-only model
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
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
# 5) Train for 10 epochs
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
# 6) Plot training curves (loss + accuracy)
# ----------------------------
def plot_training_curves(hist):
    h = hist.history
    ep = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(ep, h["loss"], label="Train Loss")
    plt.plot(ep, h["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curves"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, h["accuracy"], label="Train Acc")
    plt.plot(ep, h["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curves"); plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)

# ----------------------------
# 7) Test evaluation
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n=== TEST PERFORMANCE ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ----------------------------
# 8) Predictions + Confusion Matrix (colored)
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
plt.imshow(cm, cmap="viridis")  # colored heatmap
plt.title("Confusion Matrix (Fashion-MNIST Test) - Colored")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(range(10), class_names, rotation=35, ha="right")
plt.yticks(range(10), class_names)
plt.colorbar()

# write numbers on the heatmap for readability
thresh = cm.max() * 0.6
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)

plt.tight_layout()
plt.show()

# Print most common confusions (true -> predicted)
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
# 9) Show misclassified examples
# ----------------------------
def show_misclassified(x_test_images, y_true, y_pred, probs, n=12):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassifications found (rare).")
        return
    pick = np.random.choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)

    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    plt.figure(figsize=(12, 2.2*rows))
    for i, k in enumerate(pick):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_test_images[k], cmap="gray")
        conf = np.max(probs[k])
        plt.title(f"T={class_names[y_true[k]]}\nP={class_names[y_pred[k]]}\n({conf*100:.1f}%)")
        plt.axis("off")
    plt.suptitle("Misclassified Test Examples (Fashion-MNIST)")
    plt.tight_layout()
    plt.show()

show_misclassified(x_test_img, y_test, y_pred, all_probs, n=12)

# ----------------------------
# 10) Random test cases (visual output + confidence)
# ----------------------------
def predict_test_cases(model, x_test_flat, x_test_images, y_test, n_cases=12):
    idx = np.random.choice(len(x_test_flat), n_cases, replace=False)
    probs = model.predict(x_test_flat[idx], verbose=0)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    cols = 6
    rows = int(np.ceil(n_cases / cols))
    plt.figure(figsize=(12, 2.2*rows))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_test_images[k], cmap="gray")
        correct = (preds[i] == y_test[k])
        mark = "✓" if correct else "✗"
        plt.title(f"T={class_names[y_test[k]]}\nP={class_names[preds[i]]}\n{confs[i]*100:.1f}% {mark}")
        plt.axis("off")
    plt.suptitle("Random Test Cases: True vs Predicted + Confidence (Fashion-MNIST)")
    plt.tight_layout()
    plt.show()

    print("\nSample Predictions (index, true, pred, confidence):")
    for i, k in enumerate(idx):
        print(f"- idx={k:5d} | true={class_names[y_test[k]]:12s} | pred={class_names[preds[i]]:12s} | conf={confs[i]:.4f}")

predict_test_cases(model, x_test, x_test_img, y_test, n_cases=12)

# ----------------------------
# 11) Small detailed report
# ----------------------------
train_loss_last = history.history["loss"][-1]
val_loss_last   = history.history["val_loss"][-1]
train_acc_last  = history.history["accuracy"][-1]
val_acc_last    = history.history["val_accuracy"][-1]

print("\n" + "="*60)
print("SMALL DETAILED REPORT (Task 2: Fashion-MNIST)")
print("="*60)
print("Exploration:")
print("- Loaded Fashion-MNIST: 60,000 train and 10,000 test grayscale images (28x28).")
print("- Printed shapes/pixel ranges, plotted class distributions, and showed sample images.")
print("\nPreprocessing:")
print("- Normalized pixel values to [0, 1].")
print("- Flattened each image from 28x28 to a 784-length vector.")
print("\nModel (Dense-only):")
print("- Dense(256, ReLU) + Dropout(0.3) + Dense(128, ReLU) + Dense(10, Softmax).")
print("\nTraining (10 epochs):")
print(f"- Final Train Loss: {train_loss_last:.4f} | Final Val Loss: {val_loss_last:.4f}")
print(f"- Final Train Acc : {train_acc_last:.4f} | Final Val Acc : {val_acc_last:.4f}")
print("\nTesting:")
print(f"- Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print("\nError Analysis:")
print("- Built a colored confusion matrix (with counts in each cell).")
print("- Printed the most frequent confusions and visualized misclassified examples.")
print("="*60)
