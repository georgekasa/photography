import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def enhance_grayscale(gray_img, method='none', **kwargs):
    """
    Applies different enhancement techniques to a grayscale image.

    Args:
        gray_img (np.array): Input single-channel grayscale image.
        method (str): Enhancement method ('none', 'normalize', 'clahe', 'manual', 'threshold').
        **kwargs: Additional parameters for specific methods:
            clahe_clip (float): Clip limit for CLAHE (default: 2.0).
            clahe_grid (tuple): Tile grid size for CLAHE (default: (8, 8)).
            manual_alpha (float): Contrast control for manual method (default: 1.5).
            manual_beta (int): Brightness control for manual method (default: 0).
            threshold_val (int): Threshold value for thresholding (default: 127).

    Returns:
        np.array: Enhanced single-channel grayscale image.
    """
    print(f"Applying enhancement method: {method} with params: {kwargs}")
    if method == 'normalize':
        # Stretches contrast to full 0-255 range
        enhanced = cv2.normalize(gray_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    elif method == 'clahe':
        # Adaptive histogram equalization
        clip = kwargs.get('clahe_clip', 2.5) # Get value or default
        grid = kwargs.get('clahe_grid', (8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        enhanced = clahe.apply(gray_img)
    elif method == 'manual':
        # Manual contrast (alpha) and brightness (beta)
        alpha = kwargs.get('manual_alpha', 1.8) # Contrast
        beta = kwargs.get('manual_beta', -20)    # Brightness
        enhanced = cv2.convertScaleAbs(gray_img, alpha=alpha, beta=beta)
    elif method == 'threshold':
        # Pure black and white
        thresh_val = kwargs.get('threshold_val', 127)
        _, enhanced = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    elif method == 'none':
        # No enhancement
        enhanced = gray_img
    else:
        print(f"Warning: Unknown enhancement method '{method}'. Returning original gray.")
        enhanced = gray_img
    return enhanced

def apply_spiral_effect_and_save(
    image_path,
    output_dir,
    roi_x, roi_y, roi_w, roi_h,
    track_width=15,
    gap_width=15,
    enhancement_method='none',
    display=False,
    **enhancement_kwargs ):
    """
    Applies the spiral grayscale effect with specified enhancement and saves the result.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output image.
        roi_x, roi_y, roi_w, roi_h (int): ROI parameters.
        track_width (int): Thickness of the spiral lines.
        gap_width (int): Space between spiral lines.
        enhancement_method (str): Method passed to enhance_grayscale.
        display (bool): Whether to display the result using matplotlib.
        **enhancement_kwargs: Keyword arguments passed to enhance_grayscale.

    Returns:
        str: Path to the saved output image, or None if failed.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at path: {image_path}")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Basic color conversions
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Apply Enhancement ---
    gray_enhanced = enhance_grayscale(gray_original, enhancement_method, **enhancement_kwargs)
    gray_3ch = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR) # 3-channel for combining

    # Image dimensions & ROI clamping
    height, width = gray_original.shape
    x = max(0, roi_x); y = max(0, roi_y)
    w = min(width - x, roi_w); h = min(height - y, roi_h)

    # Create mask and draw spiral
    mask = np.zeros((height, width), dtype=np.uint8)
    left, top = x, y
    right, bottom = x + w, y + h
    while left < right and top < bottom:
        end_y_top = min(top + track_width, bottom)
        mask[top:end_y_top, left:right] = 255
        start_x_right = max(left, right - track_width)
        mask[top:bottom, start_x_right:right] = 255
        if top < end_y_top:
            start_y_bottom = max(top, bottom - track_width)
            mask[start_y_bottom:bottom, left:right] = 255
        if left < start_x_right:
            end_x_left = min(right, left + track_width)
            mask[top:bottom, left:end_x_left] = 255

        left += track_width + gap_width
        top += track_width + gap_width
        right -= track_width + gap_width
        bottom -= track_width + gap_width
        if (track_width + gap_width) <= 0 or left >= right or top >= bottom:
           break

    # Apply mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result_rgb = np.where(mask_3ch == 255, gray_3ch, image_rgb)

    # --- Save the result ---
    # Create a descriptive filename
    param_str = "_".join(f"{k}-{v}" for k, v in enhancement_kwargs.items())
    if not param_str:
        filename_base = f"spiral_{enhancement_method}"
    else:
         filename_base = f"spiral_{enhancement_method}_{param_str}"
    # Add a timestamp to avoid overwriting if run multiple times with same params
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"{filename_base}_{timestamp}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    # Convert back to BGR for saving with OpenCV
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.imwrite(output_path, result_bgr)
        print(f"Successfully saved result to: {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return None

    # --- Display if requested ---
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(result_rgb)
        plt.axis('off')
        plt.title(f'Spiral Effect - Enhancement: {enhancement_method} ({param_str})')
        plt.show()

    return output_path

# --- How to Use ---

# 1. Define your parameters
input_image = r'C:\Users\giorgosk\Pictures\fotocista_kasapoglou_004613_2025-04-29_1200\Kasapoglou_004613\Export JPG NoResize\R1-07556-023A.JPG'
output_folder = r'C:\Users\giorgosk\Pictures' # CHANGE THIS to a real folder

# ROI and Spiral settings
roi_params = {'roi_x': 435, 'roi_y': 230, 'roi_w': 900, 'roi_h': 770}
spiral_params = {'track_width': 25, 'gap_width': 25} # Adjusted gap a bit

# 2. Define the different settings you want to test
test_settings = [
    {'enhancement_method': 'none'}, # Original grayscale
    {'enhancement_method': 'normalize'},
    {'enhancement_method': 'clahe'}, # Default CLAHE
    {'enhancement_method': 'clahe', 'clahe_clip': 4.0, 'clahe_grid': (4, 4)}, # Custom CLAHE
    {'enhancement_method': 'manual', 'manual_alpha': 1.5, 'manual_beta': 0}, # Incr Contrast
    {'enhancement_method': 'manual', 'manual_alpha': 2.0, 'manual_beta': -30}, # More Contrast, Darker
    {'enhancement_method': 'threshold', 'threshold_val': 110}, # Threshold
]

# 3. Loop through settings and generate images
saved_files = []
for settings in test_settings:
    # Combine general params with specific test settings
    current_params = {**roi_params, **spiral_params, **settings}
    saved_path = apply_spiral_effect_and_save(
        image_path=input_image,
        output_dir=output_folder,
        display=False, # Set to True if you want to see each plot immediately
        **current_params # Unpacks the dictionary into keyword arguments
    )
    if saved_path:
        saved_files.append(saved_path)

print("\n--- Finished ---")
print("Generated images saved in:", output_folder)
print("Files created:")
for f in saved_files:
    print(f" - {os.path.basename(f)}")