import cv2
import numpy as np
import argparse
import sys
import os

# Global variables for polygon drawing
points = []
drawing = False
img_for_drawing = None
window_name_polygon = "Draw polygon around owl, Press ENTER when done, R to reset"

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

    cv2.namedWindow(window_name_polygon)
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


def apply_spatial_gaussian_noise(image_path, output_path, max_noise_sigma=50):
    """
    Applies Gaussian noise to an image, fading towards an object selected
    with an irregular polygon.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the noisy output image.
        max_noise_sigma (float): The maximum standard deviation of the Gaussian
                                 noise (applied furthest from the object).
    """
    # --- 1. Load Image ---
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at '{image_path}'")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image file '{image_path}'. Check format/corruption.")
        sys.exit(1)

    while True: # Loop for segmentation attempts (draw polygon -> grabcut -> confirm/reset)

        # --- 2. Select Initial Polygon (User Interaction) ---
        polygon_points = select_polygon(img)

        if polygon_points is None: # User quit during polygon selection
            print("Exiting.")
            sys.exit(0)

        # --- 3. Create Initial Mask for GrabCut ---
        # Initialize mask with definite background (GC_BGD = 0)
        mask = np.zeros(img.shape[:2], np.uint8)

        # Fill the polygon area with probable foreground (GC_PR_FGD = 3)
        # Need polygon points in the format required by fillPoly: list of arrays
        cv2.fillPoly(mask, [polygon_points], cv2.GC_PR_FGD) # Mark inside as probable foreground

        # Pixels outside the polygon remain GC_BGD (0)

        # --- 4. Segment the Owl using GrabCut (INIT_WITH_MASK) ---
        print("Running GrabCut with mask...")
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            # Apply GrabCut using the mask we created
            cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            # The 'None' rect indicates we are not using GC_INIT_WITH_RECT
        except Exception as e:
            print(f"Error during GrabCut: {e}")
            print("Please try drawing the polygon again.")
            continue # Go back to polygon selection

        # --- 5. Refine Mask and Get Binary Mask ---
        # Create a binary mask from GrabCut output:
        # where mask== GC_PR_BGD (2) or mask== GC_BGD (0) -> background -> 0
        # where mask== GC_FGD (1) or mask== GC_PR_FGD (3) -> foreground -> 1
        owl_mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # --- Visualization & Confirmation ---
        img_segmented_preview = img * owl_mask_binary[:, :, np.newaxis] # Apply mask
        preview_window_name = "Segmentation Preview (c=Confirm, r=Reset, q=Quit)"
        cv2.imshow(preview_window_name, img_segmented_preview)
        print("Showing segmentation preview.")
        print("Press 'c' to confirm, 'r' to reset (redraw polygon), 'q' to quit.")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(preview_window_name)

        if key == ord('c'):
            print("Segmentation confirmed.")
            break # Exit the loop and proceed with noise application
        elif key == ord('r'):
            print("Resetting segmentation. Draw polygon again.")
            # No need to explicitly reset mask/models, will be recreated next loop
            continue # Go back to polygon selection
        elif key == ord('q'):
            print("Quitting.")
            sys.exit(0)
        else:
            print(f"Unknown key {chr(key)}. Defaulting to confirm.")
            break

    # --- 6. Create Distance Map --- (Same as before)
    print("Calculating distance transform...")
    dist_transform = cv2.distanceTransform(1 - owl_mask_binary, cv2.DIST_L2, 5)

    # --- 7. Normalize Distance Map --- (Same as before)
    max_dist = np.max(dist_transform)
    if max_dist == 0:
        print("Warning: Max distance is 0. Is the entire image selected as the owl?")
        noise_weights = np.zeros_like(dist_transform, dtype=np.float32)
    else:
        noise_weights = (dist_transform / max_dist).astype(np.float32)

    # --- 8. Generate Gaussian Noise --- (Same as before)
    print(f"Generating Gaussian noise (max sigma={max_noise_sigma})...")
    gaussian_noise_base = np.random.normal(0, 1, img.shape).astype(np.float32)

    # --- 9. Modulate Noise Strength --- (Same as before)
    noise_weights_3channel = cv2.cvtColor(noise_weights, cv2.COLOR_GRAY2BGR)
    modulated_noise = gaussian_noise_base * max_noise_sigma * noise_weights_3channel

    # --- 10. Apply Noise to Image --- (Same as before)
    print("Applying modulated noise...")
    img_float = img.astype(np.float32)
    noisy_img_float = img_float + modulated_noise

    # --- 11. Clip and Convert Back --- (Same as before)
    noisy_img_clipped = np.clip(noisy_img_float, 0, 255)
    noisy_img_final = noisy_img_clipped.astype(np.uint8)

    # --- 12. Save Result --- (Same as before)
    print(f"Saving noisy image to '{output_path}'...")
    if not cv2.imwrite(output_path, noisy_img_final):
         print(f"Error: Could not write image to {output_path}. Check permissions/path.")
         sys.exit(1)
    print("Done!")

    # --- Optional: Display Final Result ---
    cv2.imshow("Original Image", img)
    cv2.imshow("Owl Mask (Binary)", owl_mask_binary * 255)
    cv2.imshow("Noise Weights (Normalized Distance)", noise_weights)
    cv2.imshow("Final Noisy Image", noisy_img_final)
    print("Displaying results. Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Command Line Argument Parsing --- (Same as before)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add spatially varying Gaussian noise to an image, fading towards an object selected with an irregular polygon.')
    #parser.add_argument('input_image', help='Path to the input image file (e.g., owl.jpg)')
    #parser.add_argument('output_image', help='Path to save the output noisy image (e.g., owl_noisy.jpg)')
    parser.add_argument('-s', '--sigma', type=float, default=50.0,
                        help='Maximum standard deviation (strength) of the Gaussian noise in the background (default: 50.0)')

    args = parser.parse_args()

   # apply_spatial_gaussian_noise(r'C:\Users\giorgosk\Pictures\cameras\forPost\owl.JPG', r"C:\Users\giorgosk\Pictures\cameras\owl_test1.JPG", args.sigma)
    apply_spatial_gaussian_noise(r'C:\Users\giorgosk\Pictures\cameras\forPost\owl.JPG', r"C:\Users\giorgosk\Pictures\cameras\owl_test1.JPG", 150)