import cv2
import matplotlib.pyplot as plt
import numpy as np



def load_polygon_coords(file_path):
    """Load polygon coordinates from a text file."""
    polygon = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                x, y = line[1:-1].split(',')
                polygon.append((int(x.strip()), int(y.strip())))
    return np.array(polygon, dtype=np.int32)



def create_mask_from_polygon(image_shape, polygon):
    """Create a mask from polygon coordinates."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)  # white = region to copy
    return mask

def crop_and_paste_region_effects(src_img, dst_img, mask, effect=None):
    """
    Crop a region from src_img and paste it onto dst_img using the mask.
    Supported effects: 'grayscale', 'blur', 'tint_blue', 'sepia', None
    """
    # Default: just copy the region
    region = cv2.bitwise_and(src_img, src_img, mask=mask)

    if effect == 'grayscale':
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        region = cv2.merge([gray, gray, gray])
        region = cv2.bitwise_and(region, region, mask=mask)

    elif effect == 'blur':
        blurred = cv2.GaussianBlur(src_img, (15, 15), 0)
        region = cv2.bitwise_and(blurred, blurred, mask=mask)

    elif effect == 'tint_blue':
        tint = np.full_like(src_img, (255, 0, 0))  # BGR (blue)
        region = cv2.addWeighted(region, 0.5, tint, 0.5, 0)
        region = cv2.bitwise_and(region, region, mask=mask)

    elif effect == 'sepia':
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = cv2.transform(src_img, sepia_kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        region = cv2.bitwise_and(sepia, sepia, mask=mask)

    # Create background from destination using inverse mask
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(dst_img, dst_img, mask=inv_mask)

    # Combine and return
    combined = cv2.add(background, region)
    return combined



# --- MAIN SCRIPT ---
# Load original image
image_path1 = r'C:\Users\giorgosk\Pictures\left.jpg'
img1 = cv2.imread(image_path1)
coords_file_1 = r'C:\Users\giorgosk\Documents\python\photography\segmentation_coord\left_picture_1.txt'  # your text file

image_path2 = r'C:\Users\giorgosk\Pictures\middle1.jpg'
img2 = cv2.imread(image_path2)
coords_file2 = r'C:\Users\giorgosk\Documents\python\photography\segmentation_coord\middle_picture_1.txt'  # your text file

image_path3 = r'C:\Users\giorgosk\Pictures\right.jpg'
img3 = cv2.imread(image_path3)
coords_file3 = r'C:\Users\giorgosk\Documents\python\photography\segmentation_coord\right_picture_1.txt'  # your text file

if img1.shape != img2.shape or img1.shape != img3.shape or img2.shape != img3.shape: 
    raise ValueError("Images must be the same size for direct pixel transfer")

# --- LOAD POLYGON COORDS ---

polygon1 = load_polygon_coords(coords_file_1)
polygon2 = load_polygon_coords(coords_file2)
polygon3 = load_polygon_coords(coords_file3)

mask1 = create_mask_from_polygon(img1.shape, polygon1)
mask2 = create_mask_from_polygon(img2.shape, polygon2)
mask3 = create_mask_from_polygon(img3.shape, polygon3)


# Create one base image (e.g., a copy of img2)
base = img2.copy()

#Supported effects: 'grayscale', 'blur', 'tint_blue', 'sepia', None
# Paste selected regions from img1 and img3 into base
combined1 = crop_and_paste_region_effects(img1, base, mask1, "blur")  # using grayscale=True for img1
combined2 = crop_and_paste_region_effects(img3, combined1, mask3, 'sepia')  # using updated combined1

# Show result
plt.figure(figsize=(10, 10))
final_rgb = cv2.cvtColor(combined2, cv2.COLOR_BGR2RGB)
plt.imshow(final_rgb)
plt.title("Composed image with regions from img1 and img3 into img2")
plt.axis('off')
plt.show()