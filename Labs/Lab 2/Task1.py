# ================================
# TASK 1
# ================================


import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# =======================================
# 1. Load image
# =======================================
image_path = "/content/Mosalah.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found!")

print(" Original Image:")
cv2_imshow(img)

# =======================================
# 2. Create elliptical mask
# =======================================
h, w = img.shape[:2]

# Create empty mask (all black)
mask = np.zeros((h, w), dtype=np.uint8)

# Ellipse parameters (centerX, centerY), (width, height), angle
center = (w // 2, h // 2)
axes = (w // 3, h // 2)    # bigger ellipse = more sharp area
angle = 0

# Draw white ellipse on mask
cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

print(" Elliptical Mask (White = Keep Sharp):")
cv2_imshow(mask)

# Invert mask → inside ellipse black, outside white
mask_inv = cv2.bitwise_not(mask)

print(" Inverted Mask (White = Blur Background):")
cv2_imshow(mask_inv)

# =======================================
# 3. Create strong blur
# =======================================
blurred = cv2.GaussianBlur(img, (81, 81), 0)

print(" Blurred Image:")
cv2_imshow(blurred)

# =======================================
# 4. Bitwise operations to combine
# =======================================

# Subject remains sharp (ellipse area)
sharp_subject = cv2.bitwise_and(img, img, mask=mask)

print(" Extracted Sharp Subject:")
cv2_imshow(sharp_subject)

# Background blurred (mask_inv selects outside ellipse)
blurred_background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)

print(" Extracted Blurred Background:")
cv2_imshow(blurred_background)

# Combine: OR = merges them
portrait_mode = cv2.bitwise_or(sharp_subject, blurred_background)

print(" Final Portrait Mode Effect:")
cv2_imshow(portrait_mode)

# =======================================
# 5. Save result
# =======================================
cv2.imwrite("/portrait_mode_result.jpg", portrait_mode)
print("Saved as /portrait_mode_result.jpg")
