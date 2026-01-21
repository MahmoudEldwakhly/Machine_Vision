# ================================
# TASK 2
# ================================


import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# ===================================
# 1. Load color image
# ===================================
img = cv2.imread("/content/task2-2.jpg")   # CHANGE THIS!
if img is None:
    raise ValueError("Image not found!")

print(" Original Image:")
cv2_imshow(img)

# ===================================
# 2. Make grayscale, invert, and blur
# ===================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)                    # 255-gray
blur = cv2.GaussianBlur(inverted, (25, 25), 0)      # Try (15,15), (25,25), (51,51)

print(" Grayscale Image:")
cv2_imshow(gray)

print(" Inverted Gray Image:")
cv2_imshow(inverted)

print(" Blurred Inverted Image:")
cv2_imshow(blur)

# ===================================
# 3. Color Dodge = the pencil sketch
# ===================================
sketch = cv2.divide(gray, 255 - blur, scale=256)

print(" Pencil Sketch (Color Dodge):")
cv2_imshow(sketch)

# Save result
cv2.imwrite("/pencil_sketch_colordodge.jpg", sketch)

# ===================================
# 4. Alternative Method — Edge Detection (Canny)
# ===================================
edges = cv2.Canny(gray, 50, 150)

print(" Canny Edge Sketch:")
cv2_imshow(edges)

cv2.imwrite("/pencil_sketch_edges.jpg", edges)

print("\nSaved:")
print("/pencil_sketch_colordodge.jpg")
print("/pencil_sketch_edges.jpg")
