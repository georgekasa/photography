import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
# Load image
img = cv2.imread(r'C:\Users\giorgosk\Pictures\cameras\forPost\test.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a manual mask for the hair region (placeholder - adjust this manually!)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (751, 200), (965, 375), 255, -1)  # You can refine this region

# Apply sine wave distortion only to masked region
rows, cols = img.shape[:2]
output = img.copy()

amplitude = 10*0.6
frequency = 0.08

for y in range(rows):
    offset_x = int(amplitude * np.sin(2 * np.pi * frequency * y))
    for x in range(cols):
        if mask[y, x] == 255:
            new_x = x + offset_x
            if 0 <= new_x < cols:
                output[y, x] = img[y, new_x]

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(output)
plt.title('Hair with Sine Wave Effect')
plt.axis('off')
plt.show()
