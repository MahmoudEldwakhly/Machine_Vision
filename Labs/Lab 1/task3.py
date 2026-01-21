import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# -----------------------------------------------------------
# 1. Load the two images
# -----------------------------------------------------------
img1_path = "/Mosalah_happy.jpg"   # Replace with your first image
img2_path = "/Mosalah_sad.jpg"    # Replace with your second image

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise ValueError("One or both images not found! Check file paths.")

# -----------------------------------------------------------
# 2. Resize to the same size
# -----------------------------------------------------------
height = 400
width = 400

img1_resized = cv2.resize(img1, (width, height))
img2_resized = cv2.resize(img2, (width, height))

# -----------------------------------------------------------
# 3. Create 5 cross-fade frames using addWeighted
# -----------------------------------------------------------

frames = []

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]   # 5 transition stages

for alpha in alphas:
    beta = 1 - alpha
    blended = cv2.addWeighted(img1_resized, alpha, img2_resized, beta, 0)
    frames.append(blended)

# -----------------------------------------------------------
# 4. Display the 5 frames (in Colab)
# -----------------------------------------------------------
for i, frame in enumerate(frames):
    print(f"Frame {i+1}  (alpha={alphas[i]})")
    cv2_imshow(frame)
    print("-" * 30)

# -----------------------------------------------------------
# 5. Save one final combined scene
# -----------------------------------------------------------
# Combine frames horizontally to show the transition
transition_scene = np.hstack(frames)

cv2.imwrite("/transition_scene.jpg", transition_scene)

print("Saved as /transition_scene.jpg")
