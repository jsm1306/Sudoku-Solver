import streamlit as st
import numpy as np
import cv2
from typing import Tuple, List
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('mnist_cnn.keras')

# --- Stub Functions ---
import cv2
import numpy as np
from typing import List, Optional, Tuple

def find_sudoku_grid(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the Sudoku grid in the image and return the four corner points.
    Returns None if no grid is found.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * 0.1  # At least 10% of image
    
    grid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for quadrilaterals (4 corners)
        if len(approx) == 4:
            # Check if it's roughly square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < 1.5:  # Reasonably square
                grid_contours.append((area, approx))
    
    if not grid_contours:
        return None
    
    # Take the largest qualifying contour
    _, grid_corners = max(grid_contours, key=lambda x: x[0])
    return grid_corners.reshape(4, 2).astype(np.float32)

def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left
    """
    # Sum and difference of coordinates
    sum_coords = corners.sum(axis=1)
    diff_coords = np.diff(corners, axis=1)
    
    # Top-left has smallest sum, bottom-right has largest sum
    top_left = corners[np.argmin(sum_coords)]
    bottom_right = corners[np.argmax(sum_coords)]
    
    # Top-right has smallest difference, bottom-left has largest difference
    top_right = corners[np.argmin(diff_coords)]
    bottom_left = corners[np.argmax(diff_coords)]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def process_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Detect and extract the Sudoku grid from the image.
    Apply perspective transform and split into 81 cells.
    Returns a list of 81 cell images.
    """
    original_image = image.copy()
    
    # Try to find the Sudoku grid
    grid_corners = find_sudoku_grid(image)
    
    if grid_corners is not None:
        # Order the corners properly
        ordered_corners = order_corners(grid_corners)
        
        # Define the target square (we'll use 450x450 for good resolution)
        target_size = 450
        target_corners = np.array([
            [0, 0],
            [target_size, 0],
            [target_size, target_size],
            [0, target_size]
        ], dtype=np.float32)
        
        # Apply perspective transform
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, target_corners)
        warped = cv2.warpPerspective(image, transform_matrix, (target_size, target_size))
        
        # Use the warped image for cell extraction
        grid_image = warped
        side = target_size
    else:
        # Fallback: use center square of the image
        h, w = image.shape[:2]
        side = min(h, w)
        x_start = (w - side) // 2
        y_start = (h - side) // 2
        grid_image = image[y_start:y_start+side, x_start:x_start+side]
    
    # Split into 81 cells with margin to avoid grid lines
    cell_size = side // 9
    margin = cell_size // 10  # 10% margin to avoid grid lines
    cells = []
    
    for i in range(9):
        for j in range(9):
            x = j * cell_size + margin
            y = i * cell_size + margin
            w = cell_size - 2 * margin
            h = cell_size - 2 * margin
            
            # Extract cell with margin
            cell = grid_image[y:y+h, x:x+w]
            cells.append(cell)
    
    return cells

def recognize_digits(cells: List[np.ndarray]) -> np.ndarray:
    """
    Use CNN model to recognize digits in each cell.
    Returns a 9x9 numpy matrix of digits (0 for empty).
    """
    import cv2
    import numpy as np
    from tensorflow import keras
    
    # Create a debug visualization array
    debug_images = []
    
    # Load model only once (cache)
    if not hasattr(recognize_digits, "model"):
        try:
            recognize_digits.model = keras.models.load_model("mnist_cnn.keras")
        except:
            print("Warning: Could not load model 'mnist_cnn.keras'")
            return np.zeros((9, 9), dtype=int)
    
    model = recognize_digits.model
    digits = []
    
    for i, cell in enumerate(cells):
        if cell.size == 0:  # Skip empty cells
            digits.append(0)
            continue
            
        # Create a debug image (RGB) for visualization
        debug_image = cell.copy()
        h, w = debug_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(cell.shape) == 3:
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            cell_gray = cell.copy()
        
        # Crop margin to avoid grid lines - use larger margin
        h, w = cell_gray.shape
        margin = int(min(h, w) * 0.25)  # 25% margin
        cell_cropped = cell_gray[margin:h-margin, margin:w-margin]
        
        # Apply heavy denoising first
        cell_denoised = cv2.fastNlMeansDenoising(cell_cropped, None, 10, 7, 21)
        
        # Apply Gaussian blur to further reduce noise
        cell_blurred = cv2.GaussianBlur(cell_denoised, (5, 5), 0)
        
        # Use Otsu's thresholding which is more robust for noisy images
        _, cell_thresh = cv2.threshold(cell_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Heavy morphological operations to clean noise
        # Remove small noise with opening
        kernel_open = np.ones((4, 4), np.uint8)
        cell_thresh = cv2.morphologyEx(cell_thresh, cv2.MORPH_OPEN, kernel_open)
        
        # Fill small holes with closing
        kernel_close = np.ones((3, 3), np.uint8)
        cell_thresh = cv2.morphologyEx(cell_thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove very small components (noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_thresh, connectivity=8)
        # Keep only components with reasonable size
        min_size = 20  # Minimum component size
        mask = np.zeros_like(cell_thresh)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        cell_thresh = mask
        
        # Find contours
        contours, _ = cv2.findContours(cell_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate white pixel ratio
        total_pixels = cell_thresh.shape[0] * cell_thresh.shape[1]
        white_pixels = np.sum(cell_thresh == 255)
        white_pixel_ratio = white_pixels / total_pixels
        
        # If very few white pixels, it's likely empty
        if white_pixel_ratio < 0.05 or len(contours) == 0:
            digits.append(0)
            if len(debug_image.shape) == 3:
                cv2.putText(debug_image, "Empty", (2, h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            debug_images.append(debug_image)
            continue
        
        # Filter and find the best digit contour
        valid_contours = []
        min_area = total_pixels * 0.03  # At least 10% of cell area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = max(cw, ch) / (min(cw, ch) + 1e-5)
            
            # Filter based on reasonable digit properties
            if (aspect_ratio < 4.0 and  # Not too elongated
                cw > 3 and ch > 3 and  # Not too small
                area > 20):  # Reasonable area
                valid_contours.append((area, contour))
        
        if not valid_contours:
            digits.append(0)
            if len(debug_image.shape) == 3:
                cv2.putText(debug_image, "No valid", (2, h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            debug_images.append(debug_image)
            continue
        
        # Take the largest valid contour
        _, best_contour = max(valid_contours, key=lambda x: x[0])
        x, y, cw, ch = cv2.boundingRect(best_contour)
        
        # Draw bounding box on debug image
        if len(debug_image.shape) == 3:
            cv2.rectangle(debug_image, (x, y), (x+cw, y+ch), (0, 255, 0), 1)
        
        # Extract and preprocess digit ROI
        digit_roi = cell_thresh[y:y+ch, x:x+cw]
        
        # Resize to 20x20 while maintaining aspect ratio
        digit_resized = cv2.resize(digit_roi, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28 canvas (MNIST format)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit_resized
        
        # Normalize and prepare for model
        canvas = canvas.astype("float32") / 255.0
        input_tensor = np.expand_dims(canvas, axis=(0, -1))
        
        # Predict digit
        try:
            pred = model.predict(input_tensor, verbose=0)
            digit = np.argmax(pred)
            confidence = float(pred[0][digit])
            
            # Apply confidence threshold
            if confidence < 0.5:  # Adjustable threshold
                digit = 0
                if len(debug_image.shape) == 3:
                    cv2.putText(debug_image, f"Low conf", (2, h-2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            else:
                if len(debug_image.shape) == 3:
                    cv2.putText(debug_image, f"{digit} ({confidence:.2f})", (2, h-2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            digit = 0
        
        digits.append(digit)
        debug_images.append(debug_image)
    
    # Save debug images for display
    recognize_digits.debug_images = debug_images
    
    # Convert to 9x9 matrix
    matrix = np.array(digits).reshape((9, 9))
    return matrix

# Helper function to visualize the detection process
def visualize_grid_detection(image: np.ndarray) -> np.ndarray:
    """
    Visualize the grid detection process for debugging.
    """
    vis_image = image.copy()
    
    # Find and draw the detected grid
    grid_corners = find_sudoku_grid(image)
    
    if grid_corners is not None:
        # Draw the detected grid corners
        ordered_corners = order_corners(grid_corners)
        
        # Draw the quadrilateral
        cv2.drawContours(vis_image, [ordered_corners.astype(int)], -1, (0, 255, 0), 3)
        
        # Draw corner points
        for i, corner in enumerate(ordered_corners):
            cv2.circle(vis_image, tuple(corner.astype(int)), 8, (255, 0, 0), -1)
            cv2.putText(vis_image, str(i), tuple(corner.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis_image
def solve_with_hints(matrix: np.ndarray) -> dict:
    """
    Logic-based Sudoku solver that generates step-by-step hints.
    Returns a dict with hint details (e.g., cell, value, reason).
    """
    # TODO: Implement logic-based hint generation
    pass

def overlay_hint_on_image(image: np.ndarray, hint: dict) -> np.ndarray:
    """
    Overlay hint (e.g., highlight cell, show reason) on the image using OpenCV.
    Returns the output image with overlays.
    """
    # TODO: Draw overlays for hints/mistakes
    pass

# --- Streamlit UI ---
st.set_page_config(page_title="Sudoku Tutor", layout="wide")
st.title("Sudoku Tutor")

uploaded_file = st.file_uploader("Upload a photo of a Sudoku puzzle", type=["jpg", "jpeg", "png"])

if 'board_matrix' not in st.session_state:
    st.session_state['board_matrix'] = None
if 'hint' not in st.session_state:
    st.session_state['hint'] = None
if 'original_image' not in st.session_state:
    st.session_state['original_image'] = None

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state['original_image'] = image

    # Debug: Show intermediate images for troubleshooting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    st.subheader("Step 1: Grayscale")
    st.image(gray, caption="Grayscale", use_column_width=True, channels="GRAY")
    st.subheader("Step 2: Blurred")
    st.image(blur, caption="Blurred", use_column_width=True, channels="GRAY")
    st.subheader("Step 3: Thresholded")
    st.image(thresh, caption="Thresholded", use_column_width=True, channels="GRAY")

    # Process image and recognize digits
    cells = process_image(image)
    matrix = recognize_digits(cells)
    st.session_state['board_matrix'] = matrix
    st.session_state['hint'] = None

    # Display debug cell images in grid
    if hasattr(recognize_digits, "debug_images") and len(recognize_digits.debug_images) > 0:
        st.subheader("Cell Detection Debug")
        cols = 9
        rows = 9
        for i in range(rows):
            row_cols = st.columns(cols)
            for j in range(cols):
                idx = i * cols + j
                with row_cols[j]:
                    st.image(recognize_digits.debug_images[idx], caption=f"{i+1},{j+1}", width=50)

    st.subheader("Detected Sudoku Board")
    st.write(matrix if matrix is not None else "Board not detected.")

    st.image(image, caption="Uploaded Sudoku Photo", use_column_width=True)

if st.session_state['board_matrix'] is not None:
    if st.button("Hint"):
        hint = solve_with_hints(st.session_state['board_matrix'])
        st.session_state['hint'] = hint
        st.success(f"Hint: {hint if hint else 'No hint available.'}")

    # TODO: Overlay hint on image and display
    if st.session_state['hint']:
        output_image = overlay_hint_on_image(st.session_state['original_image'], st.session_state['hint'])
        st.image(output_image, caption="Hint Overlay", use_column_width=True)

# TODO: Add support for re-uploading updated puzzle photos and comparing states
# TODO: Highlight mistakes using overlays
