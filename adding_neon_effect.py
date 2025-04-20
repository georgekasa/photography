import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Helper Function for Neon Glow Effect ---
def apply_neon_glow_to_bw(bw_image_3ch, neon_color=(255, 150, 50), glow_ksize=31, intensity_factor=0.6, highlight_bias=1.5):
    """
    Applies a neon glow effect to a 3-channel B&W image, biased towards highlights.
    (Function code is identical to the previous version - kept for completeness)
    """
    if bw_image_3ch is None or bw_image_3ch.size == 0:
        print("Warning: apply_neon_glow_to_bw received empty image.")
        # Return an empty array matching expected dtype if possible
        # It's better to let the caller handle None/empty if possible,
        # but returning an empty array might prevent immediate crashes downstream.
        # Consider the context where this function is called.
        # Returning None might be cleaner for the caller to check.
        return None # Let caller handle None case


    # Ensure kernel size is odd
    ksize = (glow_ksize, glow_ksize) if glow_ksize % 2 != 0 else (glow_ksize + 1, glow_ksize + 1)

    # Check if dimensions allow for the kernel size
    if ksize[0] > bw_image_3ch.shape[1] or ksize[1] > bw_image_3ch.shape[0]:
        print(f"Warning: Glow kernel size {ksize} is larger than the image dimensions {bw_image_3ch.shape[:2]}. Reducing kernel size.")
        ksize = (min(ksize[0], bw_image_3ch.shape[1] - (1 if bw_image_3ch.shape[1]%2==0 else 0) ), # Ensure odd and <= width
                 min(ksize[1], bw_image_3ch.shape[0] - (1 if bw_image_3ch.shape[0]%2==0 else 0) )) # Ensure odd and <= height
        # Ensure ksize remains positive
        ksize = (max(1, ksize[0]), max(1, ksize[1]))
        # Ensure ksize is odd
        ksize = (ksize[0] + (1 - ksize[0] % 2), ksize[1] + (1 - ksize[1] % 2))
        if ksize[0] <= 0 or ksize[1] <= 0:
             print("Error: Cannot apply blur with non-positive kernel size after adjustment.")
             return bw_image_3ch # Return original if kernel invalid


    # 1. Create Neon Color Layer matching the input size
    neon_layer = np.full_like(bw_image_3ch, neon_color, dtype=np.float32) / 255.0

    # 2. Blur for Glow
    try:
        neon_glow = cv2.GaussianBlur(neon_layer, ksize, 0)
    except cv2.error as e:
        print(f"Error during GaussianBlur: {e}")
        print(f"Image shape: {bw_image_3ch.shape}, ksize: {ksize}")
        return bw_image_3ch # Return original on error

    # 3. Create Intensity Map from the B&W image
    if len(bw_image_3ch.shape) == 3 and bw_image_3ch.shape[2] == 3:
        gray_intensity = cv2.cvtColor(bw_image_3ch, cv2.COLOR_BGR2GRAY)
    elif len(bw_image_3ch.shape) == 2: # Already grayscale? (Shouldn't happen based on design)
        gray_intensity = bw_image_3ch
    else:
        print("Warning: Unexpected image format in apply_neon_glow_to_bw.")
        return bw_image_3ch

    intensity_map = gray_intensity.astype(np.float32) / 255.0
    biased_intensity = np.power(intensity_map, highlight_bias)
    # Handle potential NaN/inf values from power function if highlight_bias is extreme or data has issues
    biased_intensity = np.nan_to_num(biased_intensity)
    intensity_map_3ch = cv2.cvtColor(biased_intensity, cv2.COLOR_GRAY2BGR)

    # 4. Modulate Glow with Intensity and Factor
    modulated_glow = neon_glow * intensity_map_3ch * intensity_factor

    # 5. Blend using Screen Mode Simulation
    base_float = bw_image_3ch.astype(np.float32) / 255.0
    blended_float = 1.0 - (1.0 - base_float) * (1.0 - modulated_glow)

    # Clip values and convert back to uint8
    final_image = np.clip(blended_float * 255.0, 0, 255).astype(np.uint8)
    return final_image

# --- Main Function to Apply Effect to ROI ---
def apply_effect_to_polygon(
    input_image,
    polygon_vertices, # List/array of [x, y] points defining the polygon
    neon_color=(255, 180, 255),
    glow_ksize=91,
    intensity_factor=0.2,
    highlight_bias=5.0,
    xray_tint_bgr=(180, 255, 255),
    blue_tint_color_bgr=(255, 64, 0), # Adjusted blue tint color from snippet
    tint_strength=0.15,
    final_blur_ksize=21 # Kernel size for the final blur within the polygon
    ):
    """
    Applies effects (B&W, Tint, Neon, Tint, Blur) within a specified polygon mask.

    Args:
        input_image (np.ndarray): The original BGR image.
        polygon_vertices (list or np.ndarray): A list or array of [x, y] coordinates
                                              defining the polygon vertices.
                                              e.g., [[x1, y1], [x2, y2], [x3, y3], ...]
        neon_color, glow_ksize, ... : Effect parameters (same as before).
        final_blur_ksize (int): Odd integer for the final Gaussian blur kernel size
                                applied only within the polygon area. Set to 0 or 1
                                to disable final blur.

    Returns:
        np.ndarray: The image with the effect applied within the polygon,
                    or the original image if input is invalid.
    """
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    if polygon_vertices is None or len(polygon_vertices) < 3:
        print("Error: Polygon must have at least 3 vertices.")
        return input_image # Return original if polygon is invalid

    img_h, img_w = input_image.shape[:2]
    output_image = input_image.copy() # Start with a copy of the original

    # --- 1. Create the Polygon Mask ---
    # Convert vertices to NumPy array of shape (N, 1, 2) and dtype int32
    try:
        pts = np.array(polygon_vertices, dtype=np.int32)
        # Check if vertices are within reasonable bounds (optional but good practice)
        if np.any(pts < -img_w*2) or np.any(pts > img_w*2) or \
           np.any(pts < -img_h*2) or np.any(pts > img_h*2):
            print("Warning: Some polygon vertices seem far outside image bounds.")
        pts = pts.reshape((-1, 1, 2))
    except (ValueError, TypeError) as e:
        print(f"Error converting polygon vertices: {e}. Vertices: {polygon_vertices}")
        return input_image

    # Create a black mask of the same size as the input image
    # Crucially, use a single channel (grayscale) mask for masking operations
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # Fill the polygon area in the mask with white (255)
    cv2.fillPoly(mask, [pts], 255)

    # --- 2. Determine Bounding Box of the Polygon (for efficiency) ---
    # Find min/max x and y coordinates to process only the relevant area
    x_coords = pts[:, 0, 0]
    y_coords = pts[:, 0, 1]
    x1, y1 = np.min(x_coords), np.min(y_coords)
    x2, y2 = np.max(x_coords), np.max(y_coords)

    # Clamp bounding box to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Check if bounding box has valid area
    if x1 >= x2 or y1 >= y2:
        print("Warning: Polygon bounding box has zero area after clamping. Skipping effect.")
        return input_image

    # Extract the ROI from the original image and the mask
    roi_image = input_image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2] # This mask defines the polygon *within* the ROI

    # Check if extracted roi is valid
    if roi_image.size == 0 or roi_mask.size == 0:
         print("Warning: Extracted ROI or mask is empty. Skipping effect.")
         return input_image

    # --- 3. Apply Effects Sequentially (within the ROI bounding box) ---
    # Note: We apply effects to the whole ROI rectangle first, then use the
    # roi_mask during the final blending step.

    # a) Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # b) Apply X-ray style tint
    gray_norm_roi = gray_roi.astype(np.float32) / 255.0
    xray_tint_float = np.array(xray_tint_bgr, dtype=np.float32) / 255.0
    xray_bnw_roi_float = gray_norm_roi[..., None] * xray_tint_float
    processed_bnw_roi = (xray_bnw_roi_float * 255).astype(np.uint8)

    # c) Apply Neon Glow effect
    # Apply to the B&W tinted ROI rectangle
    neon_roi = apply_neon_glow_to_bw(
        processed_bnw_roi,
        neon_color=neon_color,
        glow_ksize=glow_ksize,
        intensity_factor=intensity_factor,
        highlight_bias=highlight_bias
    )
    if neon_roi is None:
        print("Error: Neon glow function failed for ROI. Skipping subsequent steps.")
        return input_image
    # Ensure neon_roi has the same shape as roi_image in case glow func failed/returned original
    if neon_roi.shape != roi_image.shape:
         print(f"Warning: Shape mismatch after neon glow ({neon_roi.shape} vs {roi_image.shape}). Using original ROI.")
         neon_roi = processed_bnw_roi # Fallback to B&W version

    # d) Apply final Blue Tint
    blue_tint_np = np.array(blue_tint_color_bgr, dtype=np.uint8)
    blue_layer_roi = np.full_like(neon_roi, blue_tint_np)
    tinted_roi = cv2.addWeighted(
        blue_layer_roi, tint_strength, neon_roi, 1.0 - tint_strength, 0.0
    )

    # e) Apply final Gaussian Blur *only to the processed effect*
    # Ensure kernel size is odd and positive
    final_processed_roi = tinted_roi # Default if no blur
    if final_blur_ksize > 1 and final_blur_ksize % 2 != 0:
        try:
            final_processed_roi = cv2.GaussianBlur(tinted_roi, (final_blur_ksize, final_blur_ksize), 0)
        except cv2.error as e:
             print(f"Warning: Could not apply final blur with ksize {final_blur_ksize}. Error: {e}")
             final_processed_roi = tinted_roi # Use unblurred if error
    elif final_blur_ksize > 1:
        print(f"Warning: final_blur_ksize ({final_blur_ksize}) must be odd. Skipping final blur.")


    # --- 4. Blend the Processed ROI back using the Polygon Mask ---
    # Get the target area in the output image (where the ROI will be placed)
    target_roi_area = output_image[y1:y2, x1:x2]

    # Use cv2.copyTo to copy pixels from final_processed_roi to target_roi_area
    # ONLY where the roi_mask is non-zero (white).
    # This effectively blends the processed polygon area onto the original image background.
    try:
         # Ensure shapes are compatible before copyTo
         if target_roi_area.shape == final_processed_roi.shape and target_roi_area.shape[:2] == roi_mask.shape:
             cv2.copyTo(src=final_processed_roi, mask=roi_mask, dst=target_roi_area)
         else:
             print("Error: Shape mismatch during final blending. Skipping blend.")
             print(f"Target shape: {target_roi_area.shape}, Processed shape: {final_processed_roi.shape}, Mask shape: {roi_mask.shape}")
             # Don't modify output_image if shapes mismatch
    except cv2.error as e:
        print(f"Error during cv2.copyTo blending: {e}")
        # Return original image might be safest
        return input_image

    # output_image now contains the original image with the processed polygon blended in.
    return output_image

def apply_nothing(input_image, polygon_vertices):
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    if polygon_vertices is None or len(polygon_vertices) < 3:
        print("Error: Polygon must have at least 3 vertices.")
        return input_image # Return original if polygon is invalid

    img_h, img_w = input_image.shape[:2]
    output_image = input_image.copy() # Start with a copy of the original

    # --- 1. Create the Polygon Mask ---
    # Convert vertices to NumPy array of shape (N, 1, 2) and dtype int32
    try:
        pts = np.array(polygon_vertices, dtype=np.int32)
        # Check if vertices are within reasonable bounds (optional but good practice)
        if np.any(pts < -img_w*2) or np.any(pts > img_w*2) or \
           np.any(pts < -img_h*2) or np.any(pts > img_h*2):
            print("Warning: Some polygon vertices seem far outside image bounds.")
        pts = pts.reshape((-1, 1, 2))
    except (ValueError, TypeError) as e:
        print(f"Error converting polygon vertices: {e}. Vertices: {polygon_vertices}")
        return input_image

    # Create a black mask of the same size as the input image
    # Crucially, use a single channel (grayscale) mask for masking operations
    # --- END NEW SECTION ---

    # output_image now contains the original image with the B&W effect applied inside the polygon
    return input_image



# --- Function to Apply Full Effect Outside Two Polygons ---
def apply_effect_outside_polygons(
    input_image,
    poly1_vertices, # Vertices for first "original" region
    poly2_vertices, # Vertices for second "original" region
    # Effect parameters (same as before)
    neon_color=(255, 180, 255),
    glow_ksize=91,
    intensity_factor=0.2,
    highlight_bias=5.0,
    xray_tint_bgr=(180, 255, 255),
    blue_tint_color_bgr=(255, 64, 0),
    tint_strength=0.15,
    final_blur_ksize=21
    ):
    """
    Applies the full complex effect (B&W, Tint, Neon, Tint, Blur) to the area
    OUTSIDE two specified polygons, keeping the original image INSIDE them.

    Args:
        input_image (np.ndarray): Original BGR image.
        poly1_vertices (list): Vertices [[x,y],...] for the first original region.
        poly2_vertices (list): Vertices [[x,y],...] for the second original region.
        neon_color, glow_ksize, ... : Parameters for the full effect.

    Returns:
        np.ndarray: Image with effect outside polygons, original inside, or None on error.
    """
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    if poly1_vertices is None or len(poly1_vertices) < 3:
        print("Error: Polygon 1 must have at least 3 vertices.")
        return input_image
    if poly2_vertices is None or len(poly2_vertices) < 3:
        print("Error: Polygon 2 must have at least 3 vertices.")
        return input_image

    img_h, img_w = input_image.shape[:2]

    # --- 1. Calculate the Full Effect on the Entire Image ---
    # a) Convert full image to grayscale
    gray_full = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # b) Apply X-ray style tint to full grayscale
    gray_norm_full = gray_full.astype(np.float32) / 255.0
    xray_tint_float = np.array(xray_tint_bgr, dtype=np.float32) / 255.0
    xray_bnw_full_float = gray_norm_full[..., None] * xray_tint_float
    processed_bnw_full = (xray_bnw_full_float * 255).astype(np.uint8)

    # c) Apply Neon Glow effect to full tinted B&W
    neon_full = apply_neon_glow_to_bw(
        processed_bnw_full, neon_color, glow_ksize, intensity_factor, highlight_bias
    )
    if neon_full is None:
        print("Error: Neon glow failed on full image. Returning original.")
        return input_image
    # Ensure shape consistency
    if neon_full.shape != input_image.shape:
         print(f"Warning: Shape mismatch after full neon glow ({neon_full.shape} vs {input_image.shape}).")
         # Decide fallback: return original or maybe the B&W version? Let's use B&W.
         neon_full = processed_bnw_full
         if neon_full.shape != input_image.shape: # Check again
              print("Error: Fallback B&W shape also mismatch. Returning original.")
              return input_image


    # d) Apply final Blue Tint to full neon image
    blue_tint_np = np.array(blue_tint_color_bgr, dtype=np.uint8)
    blue_layer_full = np.full_like(neon_full, blue_tint_np)
    tinted_full = cv2.addWeighted(
        blue_layer_full, tint_strength, neon_full, 1.0 - tint_strength, 0.0
    )

    # e) Apply final Gaussian Blur to full tinted image
    if final_blur_ksize > 1 and final_blur_ksize % 2 != 0:
        try:
            fully_processed_image = cv2.GaussianBlur(tinted_full, (final_blur_ksize, final_blur_ksize), 0)
        except cv2.error as e:
            print(f"Warning: Final blur failed on full image: {e}. Using unblurred.")
            fully_processed_image = tinted_full
    else:
        if final_blur_ksize > 1: print(f"Warning: final_blur_ksize not odd ({final_blur_ksize}). Skipping blur.")
        fully_processed_image = tinted_full # Use unblurred if ksize is invalid

    # --- 2. Create Combined Mask for the INSIDE of Both Polygons ---
    mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
    mask2 = np.zeros((img_h, img_w), dtype=np.uint8)

    try:
        pts1 = np.array(poly1_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask1, [pts1], 255)
    except Exception as e:
        print(f"Error creating mask for Polygon 1: {e}")
        return input_image # Return original

    try:
        pts2 = np.array(poly2_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask2, [pts2], 255)
    except Exception as e:
        print(f"Error creating mask for Polygon 2: {e}")
        return input_image # Return original

    # Combine masks: Area is white if inside EITHER polygon
    combined_inside_mask = cv2.bitwise_or(mask1, mask2)

    # --- 3. Blend Original Pixels back onto the Processed Image ---
    # Where combined_inside_mask is white, copy pixels from the original input_image
    # onto the fully_processed_image.
    try:
        # Start with the fully processed image
        output_image = fully_processed_image.copy()
        # Restore original pixels inside the polygons
        cv2.copyTo(src=input_image, mask=combined_inside_mask, dst=output_image)
    except cv2.error as e:
        print(f"Error during final cv2.copyTo blending: {e}")
        # Return original image might be safest if blend fails
        return input_image

    return output_image


def apply_b_w_effect(input_image, polygon_vertices):
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    if polygon_vertices is None or len(polygon_vertices) < 3:
        print("Error: Polygon must have at least 3 vertices.")
        return input_image # Return original if polygon is invalid

    img_h, img_w = input_image.shape[:2]
    output_image = input_image.copy() # Start with a copy of the original

    # --- 1. Create the Polygon Mask ---
    # Convert vertices to NumPy array of shape (N, 1, 2) and dtype int32
    try:
        pts = np.array(polygon_vertices, dtype=np.int32)
        # Check if vertices are within reasonable bounds (optional but good practice)
        if np.any(pts < -img_w*2) or np.any(pts > img_w*2) or \
           np.any(pts < -img_h*2) or np.any(pts > img_h*2):
            print("Warning: Some polygon vertices seem far outside image bounds.")
        pts = pts.reshape((-1, 1, 2))
    except (ValueError, TypeError) as e:
        print(f"Error converting polygon vertices: {e}. Vertices: {polygon_vertices}")
        return input_image

    # Create a black mask of the same size as the input image
    # Crucially, use a single channel (grayscale) mask for masking operations
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # Fill the polygon area in the mask with white (255)
    cv2.fillPoly(mask, [pts], 255)

    # --- 2. Determine Bounding Box of the Polygon (for efficiency) ---
    # Find min/max x and y coordinates to process only the relevant area
    x_coords = pts[:, 0, 0]
    y_coords = pts[:, 0, 1]
    x1, y1 = np.min(x_coords), np.min(y_coords)
    x2, y2 = np.max(x_coords), np.max(y_coords)

    # Clamp bounding box to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Check if bounding box has valid area
    if x1 >= x2 or y1 >= y2:
        print("Warning: Polygon bounding box has zero area after clamping. Skipping effect.")
        return input_image

    # Extract the ROI from the original image and the mask
    roi_image = input_image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2] # This mask defines the polygon *within* the ROI

    # Check if extracted roi is valid
    if roi_image.size == 0 or roi_mask.size == 0:
         print("Warning: Extracted ROI or mask is empty. Skipping effect.")
         return input_image

    # --- 3. Apply Effects Sequentially (within the ROI bounding box) ---
    # Note: We apply effects to the whole ROI rectangle first, then use the
    # roi_mask during the final blending step.

    # a) Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    # --- NEW SECTION ---
    # b) Convert grayscale ROI back to 3 channels (BGR) for blending
    #    This ensures the B&W part has the same number of channels as the output image.
    gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

    # --- 4. Blend the Processed ROI back using the Polygon Mask ---
    # Get the target area in the output image (which is still the original color)
    target_roi_area = output_image[y1:y2, x1:x2]

    # Use cv2.copyTo to copy pixels from gray_roi_bgr to target_roi_area
    # ONLY where the roi_mask is non-zero (white).
    # This replaces the original color pixels inside the polygon with the B&W ones.
    try:
         # Ensure shapes are compatible before copyTo
         if target_roi_area.shape == gray_roi_bgr.shape and target_roi_area.shape[:2] == roi_mask.shape:
             cv2.copyTo(src=gray_roi_bgr, mask=roi_mask, dst=target_roi_area)
         else:
             print("Error: Shape mismatch during final blending. Skipping blend.")
             print(f"Target shape: {target_roi_area.shape}, Processed shape: {gray_roi_bgr.shape}, Mask shape: {roi_mask.shape}")
             # Return the original unmodified image in case of shape error during blend
             return input_image
    except cv2.error as e:
        print(f"Error during cv2.copyTo blending: {e}")
        # Return original image might be safest
        return input_image
    # --- END NEW SECTION ---

    # output_image now contains the original image with the B&W effect applied inside the polygon
    return output_image

def draw_polygon_callback(event, x, y, flags, param):
    """Mouse callback function for drawing a polygon."""
    global points, drawing, img_for_drawing

    img_display = param['image_display'] # Get the image passed via param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
        # print(f"Added point: ({x}, {y})") # Debug print
        # Draw points and lines on the display copy
        cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1) # Red point
        if len(points) > 1:
            cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 1) # Green line
        # Draw closing line if > 2 points
        if len(points) > 2:
             cv2.line(img_display, points[-1], points[0], (0, 255, 0), 1)
        cv2.imshow(window_name_polygon, img_display)

def select_polygon(image):
    """Allows user to draw a polygon using mouse clicks."""
    global points, drawing, img_for_drawing, window_name_polygon
    points = [] # Reset points
    drawing = False
    img_for_drawing = image.copy() # Work on a copy for drawing

    # Create a dictionary to pass parameters to the callback
    callback_param = {'image_display': img_for_drawing}




    cv2.namedWindow(window_name_polygon, cv2.WINDOW_NORMAL)
    # Pass the dictionary as the param argument
    cv2.setMouseCallback(window_name_polygon, draw_polygon_callback, param=callback_param)

    print("\n--- Draw Irregular Shape ---")
    print("1. Click points on the image to define the polygon around the owl.")
    print("2. Press ENTER/RETURN key to finalize the polygon.")
    print("3. Press 'r' key to clear points and start over.")
    print("4. Press 'q' key to quit.")

    while True:
        # Update the display in the loop in case other events happen
        # Use the img_for_drawing which is updated by the callback
        cv2.imshow(window_name_polygon, img_for_drawing)
        key = cv2.waitKey(20) & 0xFF

        if key == 13: # 13 is Enter key
            if len(points) < 3:
                print("Error: Please select at least 3 points to form a polygon.")
                continue # Keep waiting for more points or reset/quit
            else:
                print(f"Polygon defined with {len(points)} points.")
                break # Polygon finished
        elif key == ord('r'):
            print("Resetting polygon points.")
            points = []
            img_for_drawing = image.copy() # Reset drawing canvas
            # Update the param dictionary with the fresh copy
            callback_param['image_display'] = img_for_drawing
            # Redraw instructions on the fresh copy if needed (optional)
            cv2.putText(img_for_drawing, "Click points. Enter=Done, R=Reset, Q=Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name_polygon, img_for_drawing)
        elif key == ord('q'):
            print("Quitting.")
            cv2.destroyWindow(window_name_polygon)
            return None # Indicate quit

    cv2.destroyWindow(window_name_polygon)
    return np.array(points) # Return the points as a numpy array


def apply_b_w_outside_polygons(input_image, poly1_vertices, poly2_vertices):
    """
    Keeps the original image pixels inside two specified polygons and
    applies a Black & White effect to the area outside both polygons.

    Args:
        input_image (np.ndarray): The original BGR image.
        poly1_vertices (list): Vertices [[x,y],...] for the first polygon.
        poly2_vertices (list): Vertices [[x,y],...] for the second polygon.

    Returns:
        np.ndarray: Image with original colors inside polygons, B&W outside,
                    or None on error.
    """
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    if poly1_vertices is None or len(poly1_vertices) < 3:
        print("Error: Polygon 1 must have at least 3 vertices.")
        return input_image # Return original if poly1 is invalid
    if poly2_vertices is None or len(poly2_vertices) < 3:
        print("Error: Polygon 2 must have at least 3 vertices.")
        return input_image # Return original if poly2 is invalid

    img_h, img_w = input_image.shape[:2]

    # --- 1. Create the Base B&W Image ---
    # This will be the background (the "remaining" area effect)
    gray_full = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)

    # --- 2. Create Masks for Both Polygons ---
    mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
    mask2 = np.zeros((img_h, img_w), dtype=np.uint8)

    try:
        pts1 = np.array(poly1_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask1, [pts1], 255)
    except Exception as e:
        print(f"Error creating mask for Polygon 1: {e}")
        return input_image # Return original

    try:
        pts2 = np.array(poly2_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask2, [pts2], 255)
    except Exception as e:
        print(f"Error creating mask for Polygon 2: {e}")
        return input_image # Return original

    # --- 3. Combine the Masks ---
    # Create a mask where EITHER polygon 1 OR polygon 2 is defined (white).
    # This combined mask defines all areas that should remain in original color.
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # --- 4. Blend Original Pixels onto B&W Base using Combined Mask ---
    # Where the combined_mask is white, copy pixels from the original input_image
    # onto the B&W output_image.
    # This effectively restores the original color inside both polygons.
    try:
        cv2.copyTo(src=input_image, mask=combined_mask, dst=output_image)
    except cv2.error as e:
        print(f"Error during final cv2.copyTo blending: {e}")
        # Return original image might be safest if blend fails
        return input_image

    return output_image


def build_gamma_lut(gamma=1.0):
    """Builds a Look-Up Table for Gamma Correction."""
    # Ensure gamma is positive
    gamma_corr = max(0.001, gamma) # Avoid zero or negative gamma
    # Table for mapping 0-255 values
    table = np.array([((i / 255.0) ** gamma_corr) * 255.0
                      for i in np.arange(0, 256)]).astype("uint8")
    return table


def tint_background_outside_polygons(
    input_image,
    poly1_vertices,
    poly2_vertices,
    tint_type='charcoal',    # Options: 'charcoal', 'navy', 'warm_grey', 'sepia', 'gradient', 'monochrome'
    tint_strength=0.6,     # How strongly the tint color affects the base (0.0 to 1.0)
    shadow_gamma=1.0,      # Gamma for adding shadows (> 1.0 darkens/adds shadows, 1.0 = none)
    gradient_top_color=(30, 30, 30),
    gradient_bottom_color=(180, 180, 180)
    ):
    """
    Keeps the original image pixels inside two specified polygons and
    applies a color TINT to the existing background details outside those polygons.

    Args:
        input_image (np.ndarray): The original BGR image.
        poly1_vertices (list): Vertices [[x,y],...] for the first polygon.
        poly2_vertices (list): Vertices [[x,y],...] for the second polygon.
        tint_type (str): Type of tint ('charcoal', 'navy', 'warm_grey', 'sepia',
                           'gradient', 'monochrome').
        tint_strength (float): Strength of the tint overlay (0.0 = no tint, 1.0 = full color).
                               For 'gradient', controls influence of gradient color.
                               For 'monochrome', this parameter is ignored.
        gradient_top_color (tuple): BGR color for the top of the gradient tint.
        gradient_bottom_color (tuple): BGR color for the bottom of the gradient tint.

    Returns:
        np.ndarray: Image with original colors inside polygons, tinted background outside,
                    or None on error.
    """
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    # Polygon validity checks... (same as before)
    if poly1_vertices is None or len(poly1_vertices) < 3: return input_image
    if poly2_vertices is None or len(poly2_vertices) < 3: return input_image

    img_h, img_w = input_image.shape[:2]
    output_image = input_image.copy() # Start with the original

    # --- 1. Create Combined Mask for the INSIDE of Both Polygons ---
    mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
    mask2 = np.zeros((img_h, img_w), dtype=np.uint8)
    try:
        pts1 = np.array(poly1_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask1, [pts1], 255)
        pts2 = np.array(poly2_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask2, [pts2], 255)
    except Exception as e:
        print(f"Error creating masks: {e}")
        return input_image

    combined_inside_mask = cv2.bitwise_or(mask1, mask2)
    # --- Create Mask for the OUTSIDE (background area) ---
    combined_outside_mask = cv2.bitwise_not(combined_inside_mask)

    # --- 2. Prepare the Tinted Background Layer ---
    tinted_background_layer = None
    tint_type_lower = tint_type.lower()

    # --- Base layer for non-gradient tints (grayscale) ---
    # We often tint a grayscale version for cleaner results
    gray_full = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)

    if tint_type_lower == 'monochrome':
        # Just use the 3-channel grayscale image directly
        tinted_background_layer = gray_3ch

    elif tint_type_lower == 'charcoal':
        color_bgr = (54, 54, 54)
        tint_layer = np.full_like(input_image, color_bgr)
        # Blend the grayscale base with the solid charcoal color
        tinted_background_layer = cv2.addWeighted(gray_3ch, 1.0 - tint_strength, tint_layer, tint_strength, 0)

    elif tint_type_lower == 'navy':
        color_bgr = (128, 0, 0)
        tint_layer = np.full_like(input_image, color_bgr)
        # Blend the grayscale base with the solid navy color
        tinted_background_layer = cv2.addWeighted(gray_3ch, 1.0 - tint_strength, tint_layer, tint_strength, 0)

    elif tint_type_lower == 'warm_grey':
        color_bgr = (150, 150, 160)
        tint_layer = np.full_like(input_image, color_bgr)
        # Blend the grayscale base with the solid warm grey color
        tinted_background_layer = cv2.addWeighted(gray_3ch, 1.0 - tint_strength, tint_layer, tint_strength, 0)

    elif tint_type_lower == 'sepia':
         color_bgr = (112, 140, 200) # BGR Sepia-ish color
         tint_layer = np.full_like(input_image, color_bgr)
         # Blend the grayscale base with the solid sepia color
         tinted_background_layer = cv2.addWeighted(gray_3ch, 1.0 - tint_strength, tint_layer, tint_strength, 0)

    elif tint_type_lower == 'gradient':
        # Create the gradient color layer
        top_color = np.array(gradient_top_color, dtype=np.float32)
        bottom_color = np.array(gradient_bottom_color, dtype=np.float32)
        gradient_layer_float = np.zeros_like(input_image, dtype=np.float32)
        alphas = np.linspace(0, 1, img_h).reshape(-1, 1)
        row_colors = (1 - alphas) * top_color + alphas * bottom_color
        gradient_layer_float = np.tile(row_colors.reshape(img_h, 1, 3), (1, img_w, 1))
        gradient_layer = np.clip(gradient_layer_float, 0, 255).astype(np.uint8)

        # Blend the *original image* with the gradient layer
        # Here tint_strength controls how much the gradient color affects the original
        tinted_background_layer = cv2.addWeighted(input_image, 1.0 - tint_strength, gradient_layer, tint_strength, 0)

    else:
        print(f"Warning: Unknown tint_type '{tint_type}'. No tint applied to background.")
        tinted_background_layer = input_image # Use original if type is unknown


    # --- 3. Apply Shadow Effect (Gamma Correction) ---
    final_background_layer = tinted_background_layer # Start with the tinted version
    if shadow_gamma != 1.0: # Only apply if gamma is not neutral
         print(f"Applying shadow gamma: {shadow_gamma}")
         gamma_lut = build_gamma_lut(shadow_gamma)
         try:
              # Apply LUT to the tinted background layer
              final_background_layer = cv2.LUT(tinted_background_layer, gamma_lut)
         except cv2.error as e:
              print(f"Warning: Failed to apply gamma LUT: {e}. Using uncorrected tint.")
              final_background_layer = tinted_background_layer # Fallback

    # --- 4. Apply the Final Background using the OUTSIDE Mask ---
    try:
        # Copy pixels from the final_background_layer onto the output_image
        # ONLY where the combined_outside_mask is white.
        cv2.copyTo(src=final_background_layer, mask=combined_outside_mask, dst=output_image)
    except cv2.error as e:
        print(f"Error during cv2.copyTo final blend: {e}")
        return input_image # Fallback to original

    return output_image

    return output_image

def apply_background_outside_polygons(
    input_image,
    poly1_vertices,
    poly2_vertices,
    background_type='charcoal', # Options: 'charcoal', 'navy', 'warm_grey', 'gradient', 'black', 'white'
    gradient_top_color=(30, 30, 30),   # Dark Grey/Black for gradient top
    gradient_bottom_color=(180, 180, 180) # Light Grey for gradient bottom
    ):
    """
    Keeps the original image pixels inside two specified polygons and
    replaces the area outside both polygons with a specified background
    (solid color or gradient).

    Args:
        input_image (np.ndarray): The original BGR image.
        poly1_vertices (list): Vertices [[x,y],...] for the first polygon.
        poly2_vertices (list): Vertices [[x,y],...] for the second polygon.
        background_type (str): Type of background ('charcoal', 'navy',
                               'warm_grey', 'gradient', 'black', 'white').
        gradient_top_color (tuple): BGR color for the top of the gradient.
        gradient_bottom_color (tuple): BGR color for the bottom of the gradient.

    Returns:
        np.ndarray: Image with original colors inside polygons, chosen background outside,
                    or None on error.
    """
    if input_image is None or input_image.size == 0:
        print("Error: Input image is empty.")
        return None
    if poly1_vertices is None or len(poly1_vertices) < 3:
        print("Error: Polygon 1 must have at least 3 vertices.")
        return input_image # Return original if poly1 is invalid
    if poly2_vertices is None or len(poly2_vertices) < 3:
        print("Error: Polygon 2 must have at least 3 vertices.")
        return input_image # Return original if poly2 is invalid

    img_h, img_w = input_image.shape[:2]

    # --- 1. Create Combined Mask for the INSIDE of Both Polygons ---
    # (This defines the foreground area)
    mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
    mask2 = np.zeros((img_h, img_w), dtype=np.uint8)
    try:
        pts1 = np.array(poly1_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask1, [pts1], 255)
        pts2 = np.array(poly2_vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask2, [pts2], 255)
    except Exception as e:
        print(f"Error creating masks: {e}")
        return input_image

    # Combine masks: Area is white if inside EITHER polygon
    combined_inside_mask = cv2.bitwise_or(mask1, mask2)

    # --- 2. Create the Background Layer ---
    background = None
    bg_type_lower = background_type.lower() # Make comparison case-insensitive

    if bg_type_lower == 'charcoal':
        # Charcoal grey BGR value (adjust as needed)
        color_bgr = (54, 54, 54) # Example: Dark grey
        background = np.full_like(input_image, color_bgr)
    elif bg_type_lower == 'navy':
        # Deep navy blue BGR value (adjust as needed)
        color_bgr = (128, 0, 0) # Example: Classic Navy
        # color_bgr = (100, 50, 0) # Example: Slightly different navy
        background = np.full_like(input_image, color_bgr)
    elif bg_type_lower == 'warm_grey':
        # Muted warm grey BGR value (adjust as needed)
        color_bgr = (150, 150, 160) # Grey with a touch more Red
        background = np.full_like(input_image, color_bgr)
    elif bg_type_lower == 'sepia': # Added basic sepia option
         # Basic sepia simulation (convert to gray, then apply tint)
         gray_full = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
         gray_3ch = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
         # Apply sepia matrix manually (approximate) or use a library if available
         # Simple tint approximation:
         sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
         # Apply filter requires float conversion and careful handling
         # Easier way: Blend gray with a sepia color
         sepia_color = np.array([112, 140, 200], dtype=np.uint8) # BGR Sepia-ish color
         sepia_layer = np.full_like(input_image, sepia_color)
         background = cv2.addWeighted(gray_3ch, 0.6, sepia_layer, 0.4, 0) # Blend gray and sepia color
    elif bg_type_lower == 'gradient':
        # Create a vertical gradient (top darker -> bottom lighter)
        top_color = np.array(gradient_top_color, dtype=np.float32)
        bottom_color = np.array(gradient_bottom_color, dtype=np.float32)
        background = np.zeros_like(input_image, dtype=np.float32)
        # Create a ramp from 0 (top) to 1 (bottom)
        alphas = np.linspace(0, 1, img_h).reshape(-1, 1)
        # Interpolate colors for each row
        row_colors = (1 - alphas) * top_color + alphas * bottom_color
        # Tile the row colors across the width
        background = np.tile(row_colors.reshape(img_h, 1, 3), (1, img_w, 1))
        # Convert back to uint8
        background = np.clip(background, 0, 255).astype(np.uint8)
    elif bg_type_lower == 'black':
        background = np.zeros_like(input_image)
    elif bg_type_lower == 'white':
        background = np.full_like(input_image, (255, 255, 255))
    else:
        print(f"Warning: Unknown background_type '{background_type}'. Defaulting to black.")
        background = np.zeros_like(input_image)

    if background is None: # Should not happen with default case, but safety check
         print("Error: Failed to create background layer.")
         return input_image

    # --- 3. Blend Original Pixels onto the Background ---
    # Start with the generated background
    output_image = background.copy()
    # Where combined_inside_mask is white, copy pixels from the original input_image
    # onto the background.
    try:
        cv2.copyTo(src=input_image, mask=combined_inside_mask, dst=output_image)
    except cv2.error as e:
        print(f"Error during final cv2.copyTo blending: {e}")
        return input_image # Return original image might be safest if blend fails

    return output_image

# --- Main Script - Example Usage ---
if __name__ == "__main__":
# Global variables for polygon drawing
    points = []
    drawing = False
    img_for_drawing = None
    window_name_polygon = "Draw polygon around owl, Press ENTER when done, R to reset"
    # Load image
    img_path = r'C:\Users\giorgosk\Pictures\test_img3.jpg' # Use your actual path
    img = cv2.imread(img_path)

    # Handle case where image loading fails
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        exit()

    # Define the ROI (x, y, width, height)
    # Example: A rectangle in the bottom-right quadrant
    img_h, img_w = img.shape[:2]
    roi_x = 650
    roi_y = 600
    roi_w = 400
    roi_h = 1500
   # my_roi_rect =  select_polygon(img)
    my_roi_rect =  [[ 312,  593],
        [ 354, 1160],
        [ 558, 1206],
        [ 576,  885],
        [ 568,  790],
        [ 565,  742],
        [ 564,  703],
        [ 762,  695],
        [ 761,  564],
        [ 767, 490],
        [ 306,  501]]


    print(f"Applying effect to ROI: {my_roi_rect}")

    # Apply the effect using the function
    result_image = apply_nothing(
        img,
        my_roi_rect,
    )


   # my_roi_rect_2 =  select_polygon(result_image)
    my_roi_rect_2 = [
    [ 565,  712],
    [ 556, 1212],
    [ 884, 1211],
    [ 917,  501],
    [ 773,  493]]

    print(f"Applying effect to ROI: {my_roi_rect_2}")
  #  my_roi_rect_2 = []
    result_image = apply_b_w_effect(result_image,my_roi_rect_2)#einai original


    #result_image = apply_effect_outside_polygons(result_image, my_roi_rect, my_roi_rect_2)

    result_image = tint_background_outside_polygons(
                 result_image, my_roi_rect, my_roi_rect_2,
                 tint_type = 'charcoal', # Options: 'charcoal', 'navy', 'warm_grey', 'gradient', 'black', 'white'
                 tint_strength= 0.3,
                 shadow_gamma=2,
                 gradient_top_color=(60, 20, 20),    # Dark reddish top
                 gradient_bottom_color=(180, 200, 220) # Lighter beige bottom
             )




    # --- SAVE THE IMAGE ---
    if result_image is not None:
        output_filename = "roi_effect_output.jpg"
        try:
            input_dir = os.path.dirname(img_path) if os.path.dirname(img_path) else '.'
            input_filename, input_ext = os.path.splitext(os.path.basename(img_path))
            if not input_ext: input_ext = ".jpg"
            output_filename = f"{input_filename}_roi_effect{input_ext}"
            output_path = os.path.join(input_dir, output_filename)

            success = cv2.imwrite(output_path, result_image)
            if success:
                print(f"Successfully saved ROI effect image to: {output_path}")
            else:
                print(f"Error: Failed to save image to {output_path}")
        except Exception as e:
            print(f"An error occurred while saving the image: {e}")
            # Fallback save
            try:
                fallback_path = "roi_effect_output_fallback.jpg"
                print(f"Attempting to save in current directory: {fallback_path}")
                success = cv2.imwrite(fallback_path, result_image)
                if success: print(f"Saved to fallback path: {fallback_path}")
                else: print(f"Failed to save to fallback path.")
            except Exception as fallback_e:
                print(f"Failed fallback save: {fallback_e}")

        # --- Show results using Matplotlib ---
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(result_image_rgb)
        plt.title(f'Effect Applied to ROI: {my_roi_rect}')
        # Draw rectangle on plot for visualization (optional)
       # rect = plt.Rectangle((my_roi_rect[0], my_roi_rect[1]), my_roi_rect[2], my_roi_rect[3],
         #                    linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
       # plt.gca().add_patch(rect)
        plt.axis('off')
        plt.show()

    else:
        print("Image processing failed.")