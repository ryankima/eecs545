import numpy as np
import cv2
from skimage import color, morphology, filters
import time
from numba import jit
import os


# --------------------------
# Core Functions with Numba Optimization
# --------------------------

@jit(nopython=True)
def _godel_implication(x: float, y: float) -> float:
    """Gödel implication function (optimized with numba)."""
    return 1.0 if x <= y else y


@jit(nopython=True)
def _minimum_conjunction(x: float, y: float) -> float:
    """Minimum conjunction function (optimized with numba)."""
    return min(x, y)


@jit(nopython=True)
def _base_soft_color_operator(image: np.ndarray,
                              se: np.ndarray,
                              is_erosion: bool) -> np.ndarray:
    """Optimized base operator using numba."""
    h, w, c = image.shape
    se_h, se_w = se.shape
    se_center_y, se_center_x = se_h // 2, se_w // 2
    result = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            best_val = np.inf if is_erosion else -np.inf
            best_dist = np.inf
            best_channels = np.zeros(c)

            for ky in range(se_h):
                for kx in range(se_w):
                    img_y = y + ky - se_center_y
                    img_x = x + kx - se_center_x

                    if 0 <= img_y < h and 0 <= img_x < w:
                        pixel = image[img_y, img_x]
                        if np.any(np.isnan(pixel)):
                            continue

                        se_val = se[ky, kx]
                        if is_erosion:
                            aggregated = _godel_implication(se_val, pixel[0])
                        else:
                            aggregated = _minimum_conjunction(se_val, pixel[0])

                        distance = np.sqrt((ky - se_center_y) ** 2 + (kx - se_center_x) ** 2)

                        if ((is_erosion and aggregated < best_val) or
                                (not is_erosion and aggregated > best_val) or
                                (aggregated == best_val and distance < best_dist)):
                            best_val = aggregated
                            best_dist = distance
                            best_channels = pixel

            if np.isfinite(best_val):
                result[y, x, 0] = best_val
                result[y, x, 1:] = best_channels[1:]
            else:
                result[y, x] = np.nan

    return result


# --------------------------
# Pipeline Functions
# --------------------------

def load_and_preprocess(image_path: str, target_size: tuple = (600, 600)) -> np.ndarray:
    """Load and preprocess the JPG image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if needed
    if target_size and (img.shape[0] != target_size[0] or img.shape[1] != target_size[1]):
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


def remove_hairs(rgb_image: np.ndarray) -> np.ndarray:
    """Complete optimized hair removal pipeline."""
    # Convert to Lab and normalize
    lab = color.rgb2lab(rgb_image)
    lab_norm = lab.copy()
    lab_norm[..., 0] = lab[..., 0] / 100
    lab_norm[..., 1:] = (lab[..., 1:] + 128) / 255

    # Preprocess L* channel with CLAHE
    L = (lab_norm[..., 0] * 255).astype(np.uint8)
    lab_norm[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L) / 255

    # Create oriented structuring elements (reduced to 4 orientations for speed)
    angles = [0, 45, 90, 135]
    struct_elements = []
    size = 9
    center = size // 2
    for angle in angles:
        bar = np.zeros((size, size), dtype=np.uint8)
        bar[center, :] = 1
        M = cv2.getRotationMatrix2D((center, center), angle, 1)
        rotated = cv2.warpAffine(bar, M, (size, size))
        struct_elements.append(rotated)

    # Compute black top-hat for each orientation
    bth_results = []
    for se in struct_elements:
        if np.sum(se) == 0:
            continue

        # Soft color erosion and dilation
        eroded = _base_soft_color_operator(lab_norm, se, is_erosion=True)
        closed = _base_soft_color_operator(eroded, se, is_erosion=False)
        bth = closed - lab_norm
        bth_results.append(bth[..., 0])  # Only L* channel

    # Compute max and min across orientations
    max_bth = np.nanmax(bth_results, axis=0)
    min_bth = np.nanmin(bth_results, axis=0)

    # Final detector and postprocessing
    detector = max_bth - min_bth
    smoothed = filters.median(detector, morphology.disk(4))
    binary = smoothed > 0.1  # Threshold
    hair_mask = morphology.binary_dilation(binary, morphology.disk(2))

    # Inpainting
    lab_inpainted = lab_norm.copy()
    lab_inpainted[hair_mask] = np.nan

    # Create rounded structuring element for inpainting (must be uint8)
    se_size = 5
    se = np.ones((se_size, se_size), dtype=np.uint8)
    center = se_size // 2
    for i in range(se_size):
        for j in range(se_size):
            if np.sqrt((i - center) ** 2 + (j - center) ** 2) > center:
                se[i, j] = 0

    # Iterative inpainting (reduced to 5 iterations for speed)
    for _ in range(5):
        # Create temporary image where NaN = 0 and others = 255
        temp_img = np.where(np.isnan(lab_inpainted), 0, 255).astype(np.uint8)

        # For each channel
        for c in range(lab_inpainted.shape[-1]):
            channel = lab_inpainted[..., c]
            known_mask = ~np.isnan(channel)

            # Get known values
            known_values = np.where(known_mask, channel, 0)

            # Dilate and erode
            dilated = cv2.dilate(known_values.astype(np.float32), se)
            eroded = cv2.erode(np.where(known_mask, channel, 1.0).astype(np.float32), se)

            # Average
            avg = (dilated + eroded) / 2

            # Update only unknown pixels that have known neighbors
            unknown = np.isnan(channel)
            has_known_neighbor = cv2.filter2D(known_mask.astype(np.uint8), -1, np.ones((3, 3))) > 0
            update_mask = unknown & has_known_neighbor

            lab_inpainted[update_mask, c] = avg[update_mask]

    # Convert back to RGB
    lab_denorm = lab_inpainted.copy()
    lab_denorm[..., 0] = lab_denorm[..., 0] * 100
    lab_denorm[..., 1:] = lab_denorm[..., 1:] * 255 - 128
    result_rgb = color.lab2rgb(lab_denorm)

    return (result_rgb * 255).astype(np.uint8)


# --------------------------
# Main Execution
# --------------------------

def process_image(input_path: str, output_path: str = None):
    """Process a single JPG image."""
    print(f"Processing {input_path}...")
    start_time = time.time()

    try:
        # Load and preprocess
        rgb_img = load_and_preprocess(input_path)

        # Hair removal
        result = remove_hairs(rgb_img)

        # Save or return result
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"Saved result to {output_path}")

        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")
        return result

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    input_jpg = "Hairs2.jpg"  # Change to your input file
    output_jpg = "output.jpg"  # Change to desired output path

    # Process the image
    process_image(input_jpg, output_jpg)

# import cv2
# import numpy as np
# from skimage import color, morphology, filters
#
#
# def robust_hair_removal(input_path, output_path):
#     # 1. Load and preprocess image
#     img = cv2.imread(input_path)
#     if img is None:
#         raise ValueError("Could not load image")
#
#     # Convert to LAB color space
#     lab = color.rgb2lab(img)
#
#     # 2. Enhanced hair detection
#     # Work on the L channel (lightness)
#     L = lab[:, :, 0]
#
#     # Multi-scale hair detection
#     hair_mask = np.zeros_like(L, dtype=np.float32)
#     for size in [7, 9, 11]:  # Different scales for different hair thicknesses
#         for angle in [0, 45, 90, 135]:  # Multiple orientations
#             # Create oriented structuring element
#             kernel = np.zeros((size, size), dtype=np.uint8)
#             kernel[size // 2, :] = 1  # Horizontal line
#             M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
#             kernel = cv2.warpAffine(kernel, M, (size, size))
#
#             # Black top-hat transform
#             blackhat = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)
#             hair_mask = np.maximum(hair_mask, blackhat)
#
#     # Adaptive thresholding
#     _, binary_mask = cv2.threshold(hair_mask.astype(np.uint8), 0, 255,
#                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # Morphological cleanup
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#
#     # 3. Improved inpainting
#     # Prepare mask (dilate slightly to cover hair completely)
#     inpaint_mask = cv2.dilate(cleaned_mask, kernel, iterations=1)
#
#     # Convert to BGR for OpenCV inpainting
#     bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     # Use Navier-Stokes based inpainting
#     inpainted = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
#
#     # 4. Alternative post-processing (without ximgproc)
#     # Use bilateral filter for edge-preserving smoothing
#     result = cv2.bilateralFilter(inpainted, d=9, sigmaColor=75, sigmaSpace=75)
#
#     # Convert back to RGB
#     result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#
#     # Save result
#     cv2.imwrite(output_path, result_rgb)
#
#     return result_rgb
#
#
# # Example usage
# result = robust_hair_removal('Hairs2.jpg', 'improved_output.jpg')