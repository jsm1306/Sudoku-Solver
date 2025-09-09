import streamlit as st
import numpy as np
import cv2
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Load pre-trained model
model = tf.keras.models.load_model('mnist_cnn.keras')

# --- Stub Functions ---

def extract_digit_roi_improved(cell_thresh: np.ndarray) -> Optional[np.ndarray]:
    """
    Better digit ROI extraction with improved contour filtering.
    """
    # Find contours
    contours, _ = cv2.findContours(cell_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours more intelligently
    h, w = cell_thresh.shape
    min_area = (h * w) * 0.02  # At least 2% of cell area
    max_area = (h * w) * 0.8   # At most 80% of cell area
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
            
        # Get bounding rectangle
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (digits are typically taller than wide or square)
        aspect_ratio = ch / cw if cw > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 4.0:
            continue
            
        # Filter by size relative to cell
        if cw < w * 0.1 or ch < h * 0.1:  # Too small
            continue
        if cw > w * 0.9 or ch > h * 0.9:  # Too large (likely noise)
            continue
            
        valid_contours.append((area, contour, x, y, cw, ch))
    
    if not valid_contours:
        return None
    
    # Take the largest valid contour
    _, best_contour, x, y, cw, ch = max(valid_contours, key=lambda x: x[0])
    
    # Add padding around the digit (MNIST digits have some padding)
    padding_x = max(2, cw // 10)
    padding_y = max(2, ch // 10)
    
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    cw = min(w - x, cw + 2 * padding_x)
    ch = min(h - y, ch + 2 * padding_y)
    
    # Extract the ROI
    digit_roi = cell_thresh[y:y+ch, x:x+cw]
    return digit_roi

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
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * 0.1  # At least 10% of image
    
    grid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < 1.5:  # Reasonably square
                grid_contours.append((area, approx))
    
    if not grid_contours:
        return None
    
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
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    grid_corners = find_sudoku_grid(image)
    
    if grid_corners is not None:
        ordered_corners = order_corners(grid_corners)
        
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
def clean_cell(cell: np.ndarray) -> np.ndarray:
    """
    Clean and preprocess a Sudoku cell for digit recognition.
    Returns a 28x28 processed image.
    """
    # Convert to grayscale
    if cell.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    
    # Convert to grayscale
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell.copy()
    
    # Gentle noise reduction - don't over-process
    gray = cv2.medianBlur(gray, 3)  # Remove salt-and-pepper noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Light smoothing
    
    # Adaptive threshold - more conservative parameters
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4  # Larger block size, smaller C
    )
    
    # Very light morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return thresh

def resize_to_mnist_format(digit_roi: np.ndarray) -> np.ndarray:
    """
    Resize digit to 28x28 following MNIST conventions.
    MNIST digits are centered in a 28x28 field with the digit fitting in a 20x20 box.
    """
    if digit_roi.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    
    h, w = digit_roi.shape
    
    # Calculate the scaling factor to fit the digit in a 20x20 box
    # (leaving 4 pixels border on each side for centering)
    max_dim = max(h, w)
    if max_dim > 20:
        scale = 20.0 / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Resize using area interpolation (good for downsampling)
        digit_resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        digit_resized = digit_roi
        new_w, new_h = w, h
    
    # Create 28x28 canvas and center the digit
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate centering offsets
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    
    # Place digit in center
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
    
    return canvas

def recognize_digits(cells: List[np.ndarray], model, confidence_threshold: float = 0.7) -> np.ndarray:
    """
    Use CNN model to recognize digits in each cell.
    Returns a 9x9 numpy matrix of digits (0 for empty).
    """

    
    # Create a debug visualization array
    digits = []
    debug_images = []
    processed_images = []  # Store 28x28 processed images for debugging
    
    for i, cell in enumerate(cells):
        if cell.size == 0:
            digits.append(0)
            debug_images.append(np.zeros((50, 50, 3), dtype=np.uint8))
            processed_images.append(np.zeros((28, 28), dtype=np.uint8))
            continue
        
        # Step 1: Clean the cell
        cell_thresh = clean_cell(cell)
        
        # Step 2: Quick empty cell check
        white_pixels = np.sum(cell_thresh == 255)
        total_pixels = cell_thresh.shape[0] * cell_thresh.shape[1]
        white_ratio = white_pixels / total_pixels
        
        # More lenient empty cell threshold
        if white_ratio < 0.05:
            digits.append(0)
            debug_img = cell.copy() if len(cell.shape) == 3 else cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            cv2.putText(debug_img, "Empty", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            debug_images.append(debug_img)
            processed_images.append(np.zeros((28, 28), dtype=np.uint8))
            continue
        
        # Step 3: Extract digit ROI
        digit_roi = extract_digit_roi_improved(cell_thresh)
        
        if digit_roi is None:
            digits.append(0)
            debug_img = cell.copy() if len(cell.shape) == 3 else cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            cv2.putText(debug_img, "No ROI", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            debug_images.append(debug_img)
            processed_images.append(np.zeros((28, 28), dtype=np.uint8))
            continue
        
        # Step 4: Resize to 28x28 MNIST format
        mnist_image = resize_to_mnist_format(digit_roi)
        processed_images.append(mnist_image.copy())
        
        # Step 5: Normalize for the model (same as MNIST training)
        normalized = mnist_image.astype(np.float32) / 255.0
        
        # Reshape for model input: (1, 28, 28, 1)
        model_input = normalized.reshape(1, 28, 28, 1)
        
        # Step 6: Predict
        try:
            predictions = model.predict(model_input, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_digit])
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                final_digit = 0
                status_text = f"Low conf: {confidence:.2f}"
                status_color = (0, 165, 255)  # Orange
            else:
                final_digit = predicted_digit
                status_text = f"{predicted_digit} ({confidence:.2f})"
                status_color = (0, 255, 0)  # Green
                
        except Exception as e:
            print(f"Prediction error for cell {i}: {e}")
            final_digit = 0
            status_text = "Error"
            status_color = (0, 0, 255)  # Red
        
        digits.append(final_digit)
        
        # Create debug visualization
        debug_img = cell.copy() if len(cell.shape) == 3 else cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        cv2.putText(debug_img, status_text, (2, debug_img.shape[0]-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
        debug_images.append(debug_img)
    
    # Store debug info for display
    recognize_digits.debug_images = debug_images
    recognize_digits.processed_images = processed_images
    
    return np.array(digits).reshape(9, 9)

def visualize_preprocessing_steps(cell: np.ndarray) -> dict:
    """
    Visualize each step of the preprocessing pipeline for debugging.
    """
    steps = {}
    
    if cell.size == 0:
        return steps
    
    # Original
    steps['01_original'] = cell.copy()
    
    # Grayscale
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell.copy()
    steps['02_grayscale'] = gray
    
    # Noise reduction
    median_filtered = cv2.medianBlur(gray, 3)
    steps['03_median_filtered'] = median_filtered
    
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (3, 3), 0)
    steps['04_gaussian_filtered'] = gaussian_filtered
    
    # Thresholding
    thresh = cv2.adaptiveThreshold(
        gaussian_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    steps['05_thresholded'] = thresh
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps['06_morphed'] = morphed
    
    # ROI extraction
    digit_roi = extract_digit_roi_improved(morphed)
    if digit_roi is not None:
        steps['07_digit_roi'] = digit_roi
        
        # Final 28x28 image
        mnist_format = resize_to_mnist_format(digit_roi)
        steps['08_mnist_28x28'] = mnist_format
    
    return steps





# Add these new functions to your main.py file:

def detect_document_edges(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Use the original find_sudoku_grid approach which is more reliable for Sudoku grids.
    Returns the four corner points of the document/grid.
    """
    # Use the existing find_sudoku_grid function which works well for Sudoku
    return find_sudoku_grid(image)

def auto_correct_perspective(image: np.ndarray, corners: np.ndarray, output_size: int = 600) -> np.ndarray:
    """
    Apply perspective correction to straighten the document.
    Similar to Adobe Scan's auto-correction.
    """
    # Order corners: top-left, top-right, bottom-right, bottom-left
    ordered_corners = order_corners(corners)
    
    # Define destination points for a perfect square
    dst_corners = np.array([
        [0, 0],
        [output_size, 0],
        [output_size, output_size],
        [0, output_size]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
    
    # Apply perspective correction
    corrected = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))
    
    return corrected

def enhance_document_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance the document quality after perspective correction.
    Similar to document scanning apps.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to smooth while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply unsharp masking for better text clarity
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # Convert back to 3-channel for consistency
    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_color

def auto_scan_sudoku(image: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
    """
    Complete auto-scan pipeline similar to Adobe Scan.
    Returns: (corrected_image, success, debug_info)
    """
    debug_info = {}
    
    max_dim = max(image.shape[:2])
    scale_factor = 600 / max_dim
    resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Step 1: Detect document edges on resized image
    corners = detect_document_edges(resized)
    
    if corners is None:
        return image, False, {"error": "Could not detect document edges"}
    
    debug_info['detected_corners'] = corners
    
    # Scale corners back to original image size
    corners_orig_scale = corners / scale_factor
    
    # Step 2: Apply perspective correction on original image
    corrected = auto_correct_perspective(image, corners_orig_scale, output_size=600)
    debug_info['perspective_corrected'] = corrected
    
    # Step 3: Enhance document quality
    enhanced = enhance_document_quality(corrected)
    debug_info['enhanced'] = enhanced
    
    return enhanced, True, debug_info

def visualize_auto_scan_steps(image: np.ndarray, debug_info: dict) -> dict:
    """
    Create visualization images for each step of auto-scan process.
    """
    vis_images = {}
    
    # Original with detected corners
    if 'detected_corners' in debug_info:
        corners_vis = image.copy()
        corners = debug_info['detected_corners']
        
        # Draw the detected quadrilateral
        cv2.drawContours(corners_vis, [corners.astype(int)], -1, (0, 255, 0), 3)
        
        # Draw corner points
        for i, corner in enumerate(corners):
            cv2.circle(corners_vis, tuple(corner.astype(int)), 10, (255, 0, 0), -1)
            cv2.putText(corners_vis, str(i), tuple(corner.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        vis_images['corners_detected'] = corners_vis
    
    # Perspective corrected
    if 'perspective_corrected' in debug_info:
        vis_images['perspective_corrected'] = debug_info['perspective_corrected']
    
    # Enhanced
    if 'enhanced' in debug_info:
        vis_images['enhanced'] = debug_info['enhanced']
    
    return vis_images
def create_editable_sudoku_grid(matrix: np.ndarray) -> np.ndarray:
    """
    Create an editable Sudoku grid using Streamlit widgets.
    Returns the updated matrix.
    """
    st.markdown("### ✏️ Editable Sudoku Grid")
    st.markdown("*Click on any cell to edit the detected number. Use 0 or empty to clear a cell.*")



    # Create a copy to avoid modifying the original
    edited_matrix = matrix.copy()
    
    # Create the editable grid
    cols = st.columns(9)
    
    for i in range(9):
        # Add visual separation between 3x3 blocks
        if i in [3, 6]:
            st.markdown("---")
        
        with st.container():
            row_cols = st.columns(9)
            for j in range(9):
                with row_cols[j]:
                    # Create unique key for each cell
                    cell_key = f"cell_{i}_{j}"
                    
                    # Get current value (convert 0 to empty string for display)
                    current_value = matrix[i, j] if matrix[i, j] != 0 else ""
                    
                    # Create number input for each cell
                    new_value = st.text_input(
                        label="",  # No label to save space
                        value=str(current_value) if current_value != "" else "",
                        key=cell_key,
                        max_chars=1,
                        help=f"Row {i+1}, Column {j+1}",
                        label_visibility="collapsed"
                    )
                    
                    # Validate and update the matrix
                    if new_value.strip() == "":
                        edited_matrix[i, j] = 0
                    elif new_value.isdigit() and 1 <= int(new_value) <= 9:
                        edited_matrix[i, j] = int(new_value)
                    elif new_value == "0":
                        edited_matrix[i, j] = 0
                    else:
                        # Invalid input, keep original value
                        edited_matrix[i, j] = matrix[i, j]
                        if new_value.strip() != "":
                            st.error(f"Invalid input: {new_value}. Use digits 1-9 or leave empty.")
    
    return edited_matrix

def display_sudoku_comparison(original_matrix: np.ndarray, edited_matrix: np.ndarray):
    """
    Display a comparison between original detected matrix and user-edited matrix.
    """
    st.markdown("### 🔄 Changes Summary")
    
    changes = []
    for i in range(9):
        for j in range(9):
            if original_matrix[i, j] != edited_matrix[i, j]:
                # Convert to standard Python int to avoid PyArrow issues
                old_val = int(original_matrix[i, j]) if original_matrix[i, j] != 0 else "empty"
                new_val = int(edited_matrix[i, j]) if edited_matrix[i, j] != 0 else "empty"
                changes.append({
                    "Position": f"Row {i+1}, Column {j+1}",
                    "Original": str(old_val),  # Convert to string
                    "Modified": str(new_val)   # Convert to string
                })
    
    if changes:
        # Create DataFrame with explicit data types
        df_changes = pd.DataFrame(changes)
        # Ensure all columns are strings to avoid PyArrow conversion issues
        df_changes = df_changes.astype(str)
        
        # Use st.table instead of st.dataframe for better compatibility
        st.table(df_changes)
        st.success(f"✅ {len(changes)} cells modified")
    else:
        st.info("ℹ️ No changes made to the detected board")
def solve_with_hints(matrix: np.ndarray) -> dict:
    """
    Logic-based Sudoku solver that generates step-by-step hints.
    Returns a dict with hint details (e.g., cell, value, reason).
    Implements naked singles technique: find cells with only one possible candidate.
    """
    def get_candidates(r, c):
        if matrix[r, c] != 0:
            return []
        candidates = set(range(1, 10))
        # Remove digits in the same row
        candidates -= set(matrix[r, :])
        # Remove digits in the same column
        candidates -= set(matrix[:, c])
        # Remove digits in the same 3x3 block
        br, bc = 3 * (r // 3), 3 * (c // 3)
        candidates -= set(matrix[br:br+3, bc:bc+3].flatten())
        return list(candidates)

    for i in range(9):
        for j in range(9):
            if matrix[i, j] == 0:
                candidates = get_candidates(i, j)
                if len(candidates) == 1:
                    value = candidates[0]
                    reason = f"Naked single: only candidate for cell ({i+1},{j+1})"
                    return {"row": i, "col": j, "value": value, "reason": reason}
    # No naked singles found
    return {}

def overlay_hint_on_image(image: np.ndarray, hint: dict) -> np.ndarray:
    """
    Overlay hint (e.g., highlight cell, show reason) on the image using OpenCV.
    Returns the output image with overlays.
    """
    if not hint or 'row' not in hint or 'col' not in hint or 'value' not in hint:
        return image

    output = image.copy()
    h, w = output.shape[:2]

    # Calculate cell size assuming 9x9 grid and full image
    cell_w = w // 9
    cell_h = h // 9

    r = hint['row']
    c = hint['col']

    # Calculate top-left and bottom-right corners of the cell
    top_left = (c * cell_w, r * cell_h)
    bottom_right = ((c + 1) * cell_w, (r + 1) * cell_h)

    # Draw rectangle around the cell
    cv2.rectangle(output, top_left, bottom_right, (0, 0, 255), 3)

    # Put hint text above the rectangle
    text = f"Hint: {hint['value']} ({hint.get('reason', '')})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = top_left[0]
    text_y = max(top_left[1] - 10, text_size[1] + 10)

    # Draw background rectangle for text for better visibility
    cv2.rectangle(output, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)
    # Put the text
    cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return output

# --- Streamlit UI ---
st.set_page_config(
    page_title="AI Sudoku Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #64748b;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --background-color: #f8fafc;
        --card-background: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }

    /* Global styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Card styling */
    .card {
        background: var(--card-background);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
    }

    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.15);
    }

    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed var(--primary-color);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: rgba(37, 99, 235, 0.05);
        margin: 1rem 0;
    }

    /* Sudoku table styling */
    .sudoku-table {
        border-collapse: collapse;
        margin: 20px auto;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 20px;
        font-weight: 600;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .sudoku-table td {
        width: 50px;
        height: 50px;
        text-align: center;
        vertical-align: middle;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
        transition: all 0.2s ease;
    }

    .sudoku-table td:hover {
        background-color: #f1f5f9;
    }

    .sudoku-table td.thick-right {
        border-right: 3px solid var(--primary-color);
    }

    .sudoku-table td.thick-bottom {
        border-bottom: 3px solid var(--primary-color);
    }

    .sudoku-table td.thick-top {
        border-top: 3px solid var(--primary-color);
    }

    .sudoku-table td.thick-left {
        border-left: 3px solid var(--primary-color);
    }

    .sudoku-table td.empty {
        background-color: #f8fafc;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Status messages */
    .success-message {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid var(--success-color);
        color: var(--success-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }

    .info-message {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 1px solid var(--primary-color);
        color: var(--primary-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }

    .warning-message {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid var(--warning-color);
        color: var(--warning-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }

    /* Progress indicators */
    .progress-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1rem 0;
    }

    .progress-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .progress-step.active {
        background: var(--primary-color);
        color: white;
    }

    .progress-step.completed {
        background: var(--success-color);
        color: white;
    }

    .progress-step.pending {
        background: #e2e8f0;
        color: var(--text-secondary);
    }

    /* Sidebar styling */
    .sidebar-content {
        padding: 1rem;
    }

    .sidebar-section {
        background: var(--card-background);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }

    .sidebar-section h4 {
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🧠 About AI Sudoku Solver</h4>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">
            Advanced AI-powered Sudoku solver that can detect puzzles from photos and provide intelligent hints.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>📋 Features</h4>
        <ul style="color: var(--text-secondary); font-size: 0.9rem; margin: 0; padding-left: 1rem;">
            <li>📸 Photo-based puzzle detection</li>
            <li>🤖 AI digit recognition</li>
            <li>✏️ Interactive editing</li>
            <li>💡 Smart hints</li>
            <li>🔄 Reset & clear options</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>🎯 How to Use</h4>
        <ol style="color: var(--text-secondary); font-size: 0.9rem; margin: 0; padding-left: 1rem;">
            <li>Upload a clear photo of a Sudoku puzzle</li>
            <li>Review the AI-detected board</li>
            <li>Edit any incorrect cells if needed</li>
            <li>Use hints to solve step-by-step</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Sudoku Solver</h1>
    <p>Upload a photo of your Sudoku puzzle and let AI solve it for you!</p>
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">📤 Upload Your Sudoku Puzzle</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a clear photo of your Sudoku puzzle",
    type=["jpg", "jpeg", "png"],
    help="Upload a high-quality image for best results"
)

if uploaded_file:
    st.markdown('<div class="success-message">✅ File uploaded successfully! Processing...</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-message">
        💡 <strong>Tip:</strong> Take a clear, well-lit photo of your Sudoku puzzle for best results.
        Make sure all numbers are visible and the grid is not distorted.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if 'board_matrix' not in st.session_state:
    st.session_state['board_matrix'] = None
if 'hint' not in st.session_state:
    st.session_state['hint'] = None
if 'original_image' not in st.session_state:
    st.session_state['original_image'] = None
if 'initial_matrix' not in st.session_state:
    st.session_state['initial_matrix'] = None
if 'original_matrix' not in st.session_state:
    st.session_state['original_matrix'] = None


# Replace the image processing section in your Streamlit code:

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state['original_image'] = image

    # Create thresholded inverse image of original with noise reduction
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Median filter to remove salt-and-pepper noise
    denoised = cv2.medianBlur(gray, 7)
    
    # Step 2: Gaussian blur for additional smoothing
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Step 3: Adaptive threshold to create inverse image
    thresholded_inverse = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Step 4: Morphological operations to clean noise
    kernel_open = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresholded_inverse, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_dilate = np.ones((2,2), np.uint8)
    # Change dilation to erosion to reduce digit thickness and improve recognition
    final_cleaned = cv2.erode(closed, kernel_dilate, iterations=1)
    
    # Add debug images before and after erosion to verify effect
    st.subheader("📋 Thresholded Inverse Image Processing")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image(gray, caption="1. Original Grayscale", use_column_width=True, channels="GRAY")
        st.image(denoised, caption="2. Median Filtered", use_column_width=True, channels="GRAY")
    
    with col2:
        st.image(blurred, caption="3. Gaussian Blurred", use_column_width=True, channels="GRAY")
        st.image(thresholded_inverse, caption="4. Thresholded Inverse", use_column_width=True, channels="GRAY")
    
    with col3:
        st.image(closed, caption="5. Cleaned (Open+Close)", use_column_width=True, channels="GRAY")
    
    with col4:
        st.image(final_cleaned, caption="6. Final (Eroded)", use_column_width=True, channels="GRAY")

    st.subheader("🔍 Auto-Scanning Sudoku Grid...")
    
    corrected_image, scan_success, debug_info = auto_scan_sudoku(image)
    
    if scan_success:
        st.success("✅ Sudoku grid detected and corrected successfully!")
        
        # Show auto-scan steps
        vis_images = visualize_auto_scan_steps(image, debug_info)
        
        st.subheader("Auto-Scan Steps")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption="1. Original Image", use_column_width=True)
            if 'corners_detected' in vis_images:
                st.image(vis_images['corners_detected'], caption="2. Detected Edges", use_column_width=True)
        
        with col2:
            if 'perspective_corrected' in vis_images:
                st.image(vis_images['perspective_corrected'], caption="3. Perspective Corrected", use_column_width=True)
        
        with col3:
            if 'enhanced' in vis_images:
                st.image(vis_images['enhanced'], caption="4. Enhanced Quality", use_column_width=True)
        
        process_image_input = corrected_image
        
    else:
        st.warning("⚠️ Could not auto-detect grid. Using original image.")
        st.write(f"Error: {debug_info.get('error', 'Unknown error')}")
        process_image_input = image
    
    cells = process_image(process_image_input)
    matrix = recognize_digits(cells, model, confidence_threshold=0.6)
    st.session_state['board_matrix'] = matrix
    st.session_state['hint'] = None

    if hasattr(recognize_digits, "processed_images"):
        st.subheader("28x28 Images Sent to CNN Model")
        cols = st.columns(9)
        for i in range(9):
            for j in range(9):
                idx = i * 9 + j
                with cols[j]:
                    if idx < len(recognize_digits.processed_images):
                        img_28x28 = recognize_digits.processed_images[idx]
                        display_img = (img_28x28 * 255).astype(np.uint8) if img_28x28.max() <= 1.0 else img_28x28
                        st.image(display_img, caption=f"({i+1},{j+1})", width=50, clamp=True)
            st.write("")  

    if hasattr(recognize_digits, "debug_images"):
        st.subheader("Cell Detection Results")
        for i in range(9):
            cols = st.columns(9)
            for j in range(9):
                idx = i * 9 + j
                with cols[j]:
                    if idx < len(recognize_digits.debug_images):
                        st.image(recognize_digits.debug_images[idx], 
                               caption=f"R{i+1}C{j+1}: {matrix[i,j]}", width=50)

    # Add debugging tools
    # st.subheader("🔧 Debugging Tools")
    
    # # Cell-by-cell debugging
    
    
    # # Confidence threshold adjustment
    # st.subheader("⚙️ Model Parameters")
    # new_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    
    # if st.button("Re-run Recognition with New Threshold"):
    #     matrix = recognize_digits(cells, model, confidence_threshold=new_threshold)
    #     st.session_state['board_matrix'] = matrix
    #     st.rerun()


    st.subheader("🎯 Detected Sudoku Board")
    if matrix is not None:
        if 'original_matrix' not in st.session_state:
            st.session_state['original_matrix'] = matrix.copy()
        if st.session_state['initial_matrix'] is None:
            st.session_state['initial_matrix'] = matrix.copy()
        
        st.markdown("### 🤖 AI Detected Board (Read-Only)")
        
        html_table = """
        <style>
        .sudoku-table {
            border-collapse: collapse;
            margin: 20px auto;
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
        }
        .sudoku-table td {
            width: 40px;
            height: 40px;
            text-align: center;
            vertical-align: middle;
            border: 1px solid #666;
            background-color: #f9f9f9;
        }
        .sudoku-table td.thick-right {
            border-right: 3px solid #000;
        }
        .sudoku-table td.thick-bottom {
            border-bottom: 3px solid #000;
        }
        .sudoku-table td.thick-top {
            border-top: 3px solid #000;
        }
        .sudoku-table td.thick-left {
            border-left: 3px solid #000;
        }
        .sudoku-table td.empty {
            background-color: #fff;
            color: #ccc;
        }
        </style>
        <table class="sudoku-table">
        """
        
        for i in range(9):
            html_table += "<tr>"
            for j in range(9):
                # Determine cell classes for thick borders
                classes = []
                if j in [2, 5]:  # Right border of 3x3 blocks
                    classes.append("thick-right")
                if i in [2, 5]:  # Bottom border of 3x3 blocks  
                    classes.append("thick-bottom")
                if i == 0:  # Top border
                    classes.append("thick-top")
                if j == 0:  # Left border
                    classes.append("thick-left")
                if j == 8:  # Right border
                    classes.append("thick-right")
                if i == 8:  # Bottom border
                    classes.append("thick-bottom")
                
                # Get cell value
                cell_value = matrix[i, j]
                if cell_value == 0:
                    classes.append("empty")
                    display_value = ""
                else:
                    display_value = str(cell_value)
                
                class_str = ' class="' + ' '.join(classes) + '"' if classes else ''
                html_table += f'<td{class_str}>{display_value}</td>'
            
            html_table += "</tr>"
        
        html_table += "</table>"
        
        # Display the HTML table
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Create editable version
        edited_matrix = create_editable_sudoku_grid(matrix)
        
        # Update session state with edited matrix
        st.session_state['board_matrix'] = edited_matrix
        
        # Show comparison if there are changes
        if st.session_state['original_matrix'] is not None:
            display_sudoku_comparison(st.session_state['original_matrix'], edited_matrix)
        
        # Also show a simple text representation of current state
        st.markdown("### 📝 Current Board State (Text)")
        text_repr = ""
        for i in range(9):
            if i in [3, 6]:
                text_repr += "------+-------+------\n"
            for j in range(9):
                if j in [3, 6]:
                    text_repr += "| "
                cell_value = edited_matrix[i, j]
                text_repr += str(cell_value) if cell_value != 0 else "."
                text_repr += " "
            text_repr += "\n"
        
        st.code(text_repr, language=None)
        
    else:
        st.error("❌ Board not detected. Please try with a clearer image.")

if st.session_state['board_matrix'] is not None:
    if st.button("💡 Get Hint"):
        hint = solve_with_hints(st.session_state['board_matrix'])
        st.session_state['hint'] = hint
        st.success(f"Hint: {hint if hint else 'No hint available.'}")

    if st.session_state['hint']:
        output_image = overlay_hint_on_image(st.session_state['original_image'], st.session_state['hint'])
        st.image(output_image, caption="Hint Overlay", use_column_width=True)