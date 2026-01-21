# ===============================
# Digits Dataset By Mahmoud Elsayd - 21P0017
# ===============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. Load and Explore the Dataset
# -------------------------------
digits = load_digits()

print("Dataset keys:", digits.keys())
print("Data shape:", digits.data.shape)
print("Images shape:", digits.images.shape)
print("Target shape:", digits.target.shape)
print("Unique labels:", np.unique(digits.target))

# -------------------------------
# 2. Show the First 50 Images
# -------------------------------
fig, axes = plt.subplots(5, 10, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(digits.target[i])
    ax.axis('off')

plt.suptitle("First 50 Images from Digits Dataset", fontsize=14)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 3. Train-Test Split (test size = 0.25)
# --------------------------------------------
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# -------------------------------
# 4. Standardization
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. Train KNN (k = 3)
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Predictions & Accuracy
# -------------------------------
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy (k=3): {accuracy:.4f}")

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(10))
plt.yticks(range(10))

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.show()

# -------------------------------
# 8. Classification Report
# -------------------------------
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
