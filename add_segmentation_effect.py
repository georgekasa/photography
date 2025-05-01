import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_polygon_coords(file_path):
    """Load polygon coordinates from a text file.
    Expects lines like: (x, y)
    """
    polygon = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    try:
                        x, y = line[1:-1].split(',')
                        polygon.append((int(x.strip()), int(y.strip())))
                    except ValueError:
                        logging.warning(f"Could not parse line: {line} in {file_path}")
        if not polygon:
            logging.warning(f"No valid coordinates found in {file_path}")
            return None
        return np.array(polygon, dtype=np.int32)
    except FileNotFoundError:
        logging.error(f"Coordinate file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading coordinate file {file_path}: {e}")
        return None

def scale_polygon(polygon, scale_factor):
    """
    Scale a polygon's area up (>1) or down (<1) relative to its centroid.

    Args:
        polygon (np.ndarray): Nx2 array of (x, y) coordinates
        scale_factor (float): e.g., 1.2 for +20%, 0.8 for -20%

    Returns:
        np.ndarray: Scaled Nx2 polygon (int coordinates)
    """
    if polygon is None or len(polygon) < 3:
         logging.error("Cannot scale polygon: Invalid input.")
         return None
    # Compute centroid
    centroid = polygon.mean(axis=0)

    # Shift polygon to origin (centroid becomes [0, 0])
    shifted = polygon - centroid

    # Scale
    scaled = shifted * scale_factor

    # Shift back
    result = scaled + centroid

    return result.astype(np.int32) # Use np.int32 for OpenCV compatibility

def create_mask_from_polygon(image_shape, polygon):
    """Create a binary mask from polygon coordinates."""
    if polygon is None or len(polygon) < 3:
         logging.error("Cannot create mask: Invalid polygon.")
         return None
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # Ensure polygon is in the correct format for fillPoly (list of arrays)
    cv2.fillPoly(mask, [polygon.reshape((-1, 1, 2))], 255) # white = region of interest
    return mask

def enhance_grayscale(gray_img, method='none', **kwargs):
    """
    Applies different enhancement techniques to a grayscale image.

    Args:
        gray_img (np.array): Input single-channel grayscale image.
        method (str): Enhancement method ('none', 'normalize', 'clahe', 'manual', 'threshold').
        **kwargs: Additional parameters for specific methods:
            clahe_clip (float): Clip limit for CLAHE (default: 2.0).
            clahe_grid (tuple): Tile grid size for CLAHE (default: (8, 8)).
            manual_alpha (float): Contrast control (default: 1.5).
            manual_beta (int): Brightness control (default: 0).
            threshold_val (int): Threshold value (default: 127).

    Returns:
        np.array: Enhanced single-channel grayscale image.
    """
    logging.info(f"Applying enhancement method: {method} with params: {kwargs}")
    if method == 'normalize':
        enhanced = cv2.normalize(gray_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    elif method == 'clahe':
        clip = kwargs.get('clahe_clip', 2.0)
        grid = kwargs.get('clahe_grid', (8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        enhanced = clahe.apply(gray_img)
    elif method == 'manual':
        alpha = kwargs.get('manual_alpha', 1.5) # Contrast
        beta = kwargs.get('manual_beta', 0)    # Brightness
        enhanced = cv2.convertScaleAbs(gray_img, alpha=alpha, beta=beta)
    elif method == 'threshold':
        thresh_val = kwargs.get('threshold_val', 127)
        _, enhanced = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    elif method == 'none':
        enhanced = gray_img.copy() # Return a copy to avoid modifying original
    else:
        logging.warning(f"Unknown enhancement method '{method}'. Returning original gray.")
        enhanced = gray_img.copy()
    return enhanced

def apply_enhancement_to_region(original_bgr_img, mask, enhancement_method='none', **enhancement_params):
    """
    Applies grayscale enhancement only to the masked region of an image.

    Args:
        original_bgr_img (np.array): The original BGR image.
        mask (np.array): Binary mask (uint8), where 255 indicates the region to enhance.
        enhancement_method (str): The method passed to enhance_grayscale.
        **enhancement_params: Keyword arguments passed to enhance_grayscale.

    Returns:
        np.array: BGR image with the specified region enhanced.
    """
    if mask is None:
        logging.error("Cannot apply enhancement: Invalid mask provided.")
        return original_bgr_img.copy() # Return original if mask failed

    # 1. Convert the *entire* original image to grayscale for consistent processing
    gray_img = cv2.cvtColor(original_bgr_img, cv2.COLOR_BGR2GRAY)

    # 2. Apply the enhancement to the grayscale image
    enhanced_gray = enhance_grayscale(gray_img, enhancement_method, **enhancement_params)

    # 3. Convert the enhanced grayscale back to 3 channels (BGR)
    #    We need this to combine it with the original color background
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # 4. Extract the enhanced region using the mask
    enhanced_region = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=mask)

    # 5. Extract the original background (outside the mask)
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(original_bgr_img, original_bgr_img, mask=inv_mask)

    # 6. Combine the background and the enhanced region
    result_img = cv2.add(background, enhanced_region)

    return result_img

def create_output_filename(base_filename, settings):
    """Creates a descriptive filename based on enhancement settings."""
    name_parts = [os.path.splitext(base_filename)[0]] # Start with original name without extension
    method = settings.get('enhancement_method', 'unknown')
    name_parts.append(method)

    # Add specific parameters to the filename for clarity
    if method == 'clahe':
        clip = settings.get('clahe_clip')
        grid = settings.get('clahe_grid')
        if clip is not None: name_parts.append(f"clip{clip}")
        if grid is not None: name_parts.append(f"grid{grid[0]}x{grid[1]}")
    elif method == 'manual':
        alpha = settings.get('manual_alpha')
        beta = settings.get('manual_beta')
        if alpha is not None: name_parts.append(f"alpha{alpha}")
        if beta is not None: name_parts.append(f"beta{beta}")
    elif method == 'threshold':
        thresh = settings.get('threshold_val')
        if thresh is not None: name_parts.append(f"thresh{thresh}")

    return "_".join(map(str, name_parts)) + ".jpg" # Join with underscores


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # --- Input Parameters ---
    image_path = r'c:\Users\giorgosk\Pictures\fotocista_kasapoglou_004613_2025-04-29_1200\Kasapoglou_004613\Export JPG NoResize\R1-07556-015A.JPG'
    coords_file = r'photography/segmentation_coord/detection_results_R1-07556-015A.txt'
    output_folder = r'c:\Users\giorgosk\Pictures\fotocista_kasapoglou_004613_2025-04-29_1200\Kasapoglou_004613' # Define output directory
    polygon_scale_factor = 1.3

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # --- Load Data ---
    logging.info(f"Loading image: {image_path}")
    img_original = cv2.imread(image_path)
    if img_original is None:
        logging.error(f"Failed to load image: {image_path}. Exiting.")
        exit()

    logging.info(f"Loading polygon coordinates: {coords_file}")
    polygon_original = load_polygon_coords(coords_file)
    if polygon_original is None:
        logging.error("Failed to load polygon coordinates. Exiting.")
        exit()

    # --- Prepare Mask ---
    logging.info(f"Scaling polygon by factor: {polygon_scale_factor}")
    polygon_scaled = scale_polygon(polygon_original, polygon_scale_factor)
    if polygon_scaled is None:
         logging.error("Failed to scale polygon. Exiting.")
         exit()

    logging.info("Creating mask from scaled polygon")
    mask = create_mask_from_polygon(img_original.shape, polygon_scaled)
    if mask is None:
        logging.error("Failed to create mask. Exiting.")
        exit()

    # --- Define Enhancement Settings ---
    test_settings = [
        {'enhancement_method': 'none'}, # Original grayscale in region
        {'enhancement_method': 'normalize'},
        {'enhancement_method': 'clahe'}, # Default CLAHE
        {'enhancement_method': 'clahe', 'clahe_clip': 4.0, 'clahe_grid': (4, 4)}, # Custom CLAHE
        {'enhancement_method': 'manual', 'manual_alpha': 1.5, 'manual_beta': 0}, # Incr Contrast
        {'enhancement_method': 'manual', 'manual_alpha': 2.0, 'manual_beta': -30}, # More Contrast, Darker
        {'enhancement_method': 'threshold', 'threshold_val': 110}, # Threshold
    ]

    # --- Process and Save ---
    saved_files = []
    final_results_for_display = [] # Optional: collect results for plotting

    base_image_filename = os.path.basename(image_path)

    for i, settings in enumerate(test_settings):
        logging.info(f"\nProcessing setting {i+1}/{len(test_settings)}: {settings}")

# Apply the enhancement to the specified region
        # REMOVE the explicit enhancement_method=... keyword argument here
        result_img = apply_enhancement_to_region(
            img_original,
            mask,
            **settings  # Let dictionary unpacking handle all keyword arguments
        )

        # Generate filename and save
        output_filename = create_output_filename(base_image_filename, settings)
        save_path = os.path.join(output_folder, output_filename)

        try:
            cv2.imwrite(save_path, result_img)
            logging.info(f"Successfully saved: {save_path}")
            saved_files.append(save_path)
            final_results_for_display.append(result_img) # Add for potential display
        except Exception as e:
            logging.error(f"Failed to save {save_path}: {e}")


    print("\n--- Finished ---")
    print(f"Generated images saved in: {os.path.abspath(output_folder)}")
    print("Files created:")
    for f in saved_files:
        print(f" - {os.path.basename(f)}")

    # --- Optional: Display results using Matplotlib ---
    if final_results_for_display:
        num_results = len(final_results_for_display)
        cols = 3 # Adjust number of columns as needed
        rows = (num_results + cols - 1) // cols
        plt.figure(figsize=(cols * 5, rows * 5)) # Adjust figure size

        for i, (img, settings) in enumerate(zip(final_results_for_display, test_settings)):
            plt.subplot(rows, cols, i + 1)
            # Construct title from settings for clarity
            title = settings.get('enhancement_method', 'none')
            params_str = ', '.join(f"{k}={v}" for k, v in settings.items() if k != 'enhancement_method')
            if params_str:
                title += f" ({params_str})"

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()