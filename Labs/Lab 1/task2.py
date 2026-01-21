import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# ==========================================
# TASK 2 — Social Media Themed Photo Frame
# ==========================================

# 1. Load image
image_path = "/Mosalah.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found!")

print(" Original Image:")
cv2_imshow(img)

# 2. Flip horizontally
flipped = cv2.flip(img, 1)
print(" Flipped Image (Selfie Effect):")
cv2_imshow(flipped)

# 3. Create thick rectangle frame simulation
frame_img = flipped.copy()
h, w, c = frame_img.shape

color = (0, 120, 255)  # orange BGR
thickness = 25

# Draw top, bottom, left, right with margin to simulate round corners
cv2.line(frame_img, (40, 20), (w-40, 20), color, thickness)
cv2.line(frame_img, (40, h-20), (w-40, h-20), color, thickness)
cv2.line(frame_img, (20, 40), (20, h-40), color, thickness)
cv2.line(frame_img, (w-20, 40), (w-20, h-40), color, thickness)

print(" Final Image with Rounded-Style Frame:")
cv2_imshow(frame_img)

# 4. Save
cv2.imwrite("/profile_frame.jpg", frame_img)
print("Saved as: /profile_frame.jpg")
