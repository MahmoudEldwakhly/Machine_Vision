# ================================
# TASK 1
# ================================


import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# ============================================
# 1. Load image
# ============================================
img = cv2.imread("/content/Task1.jpg")   # <-- change file name
if img is None:
    raise ValueError("Image not found!")

print("Original Image:")
cv2_imshow(img)

# ============================================
# 2. Convert to grayscale
# ============================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Grayscale Image:")
cv2_imshow(gray)

# ============================================
# 3. Histogram Equalization (Contrast Enhancement)
# ============================================
equalized = cv2.equalizeHist(gray)
print("After Histogram Equalization:")
cv2_imshow(equalized)

# ============================================
# 4. Edge Detection — Laplacian
# ============================================
laplacian = cv2.Laplacian(equalized, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)  # convert float → uint8

print("Laplacian Edges:")
cv2_imshow(laplacian)

# ============================================
# 5. Edge Detection — Canny
# ============================================
canny = cv2.Canny(equalized, 50, 150)

print("Canny Edges:")
cv2_imshow(canny)

# ============================================
# 6. Artistic blending — soften edges + combine
# ============================================

# soften edges
lap_blur = cv2.GaussianBlur(laplacian, (7, 7), 0)
canny_blur = cv2.GaussianBlur(canny, (7, 7), 0)

# combine edges (weight them)
combined_edges = cv2.addWeighted(lap_blur, 0.6, canny_blur, 0.4, 0)

# invert edges for sketch look
edges_inv = cv2.bitwise_not(combined_edges)

print("Combined Soft Edges (Inverted):")
cv2_imshow(edges_inv)

# convert original image to grayscale for blending
gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# make into 3-channel for blending
edges_color = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

# blend with original
artistic_sketch = cv2.addWeighted(img, 0.5, edges_color, 0.5, 0)

print("Final Artistic Sketch-Style Image:")
cv2_imshow(artistic_sketch)

# save result
cv2.imwrite("/artistic_sketch_result.jpg", artistic_sketch)
print("Saved as /artistic_sketch_result.jpg")


