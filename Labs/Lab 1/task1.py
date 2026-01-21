import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# ================================
# TASK 1 — Character ID Card
# ================================

# 1. Load image
image_path = "/Mosalah.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found!")

print(" Original Image:")
cv2_imshow(img)

# 2. Resize
photo = cv2.resize(img, (200, 200))
print(" Resized 200 x 200:")
cv2_imshow(photo)

# 3. Create white background
card_width = 400
card_height = 300
card = np.ones((card_height, card_width, 3), dtype=np.uint8) * 255
print(" Blank White ID Card:")
cv2_imshow(card)

# 4. Insert the photo
card[50:250, 20:220] = photo
print(" After Inserting Photo:")
cv2_imshow(card)

# 5. Add text
cv2.putText(card, "Mo Salah ", (230, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv2.putText(card, "Liverpool", (230, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 2)

cv2.putText(card, "ID: 11 - 10", (230, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)

print(" After Adding Text:")
cv2_imshow(card)

# 6. Add border
bordered = cv2.copyMakeBorder(card, 5, 5, 5, 5,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
print(" Final Bordered ID Card:")
cv2_imshow(bordered)

# 7. Save
cv2.imwrite("/character_id_card.jpg", bordered)
print("Saved as: /character_id_card.jpg")
