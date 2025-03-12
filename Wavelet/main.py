from src import wavelet_segmentation, display_results
from src.config import DATASET_PATH
from src.image_processing import load_image
import os

def load_all_images(folder_path):
    """
    Load all images from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
    
    Returns:
        list: A list of tuples containing the loaded image and its filename.
    """
    images = []
    # Supported image file extensions
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a valid image extension
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(folder_path, filename)
            img = load_image(image_path)
            images.append((img, filename))
        else:
            print(f"Skipping non-image file: {filename}")
    
    return images

if __name__ == "__main__":
    # Load all images from the dataset folder
    all_images = load_all_images("testing")
    
    # Process and display each image
    for img, img_file in all_images:
        binary_mask, hair_removed_img = wavelet_segmentation(img)
        display_results(img, binary_mask, hair_removed_img, img_file)