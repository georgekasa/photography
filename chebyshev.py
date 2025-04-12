import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from scipy.special import eval_chebyt  # Use Chebyshev polynomials
from scipy.special import eval_chebyc  # Use Chebyshev polynomials
from scipy.special import eval_chebys  # Use Chebyshev polynomials
from scipy.special import eval_chebyu  # Use Chebyshev polynomials
# Load and prep image
img = cv2.imread(r'C:\Users\giorgosk\Pictures\cameras\forPost\test.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols = img.shape[:2]
output = img.copy()

# Define TWO rectangular areas: (x1, y1, x2, y2)
rectangles = [
    (0, 0, 500, 1225),  # Rectangle A
    (1200, 0, 1810, 1225)   # Rectangle B
]

# Legendre settings
degree = 17
amplitude = 20

for rect in rectangles:
    x1, y1, x2, y2 = rect
    box_height = y2 - y1

    # Generate offsets for this rectangle using Legendre
    y_vals = np.linspace(-1, 1, box_height)
    offsets = amplitude * eval_chebyu(degree, y_vals)

    # Apply distortion inside the rectangle
    for dy, y in enumerate(range(y1, y2)):
        offset = int(offsets[dy])
        for x in range(x1, x2):
            new_x = x + offset
            if 0 <= new_x < cols:
                output[y, x] = img[y, new_x]

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(output)
plt.title('Legendre Distortion in Two Rectangles')
plt.axis('off')
plt.show()

#cv2.imwrite('distorted_image.jpg', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
cv2.imwrite(r'C:\Users\giorgosk\Pictures\cameras\forPost\distorted_image.jpg', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
