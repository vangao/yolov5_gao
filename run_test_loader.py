import cv2
import numpy as np
import torch
import os
import yaml # For dummy hyp

# This script is intended to be run in an environment where 
# OpenCV, NumPy, PyYAML, and PyTorch are correctly installed,
# and from the root directory of the YOLOv5 repository.

print(f"Attempting to use cv2 version: {cv2.__version__}")
print(f"Attempting to use numpy version: {np.__version__}")
print(f"Attempting to use torch version: {torch.__version__}")

# Ensure utils can be imported if script is in root
import sys
# Corrected sys.path to point to the root for utils.dataloaders
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 

from utils.dataloaders import LoadImagesAndLabels
# from utils.augmentations import Albumentations # Not strictly needed for this test

print("Starting 12-bit loader test script...")

# Create a dummy hyp dict (minimal, focus on loading and basic augmentations)
# Set HSV gains to 0 to avoid color changes if the image is mostly one color,
# which could make max value checks less reliable if not handled carefully.
hyp = {
    'mosaic': 0.0, 'mixup': 0.0, 'degrees': 0.0, 'translate': 0.0, 
    'scale': 0.0, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 
    'fliplr': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 
    'copy_paste': 0.0
}

# Define paths relative to the script execution directory (assumed to be YOLOv5 root)
sample_image_filename = 'sample_12bit.tif'
sample_image_path = os.path.join('data', 'images', sample_image_filename)
label_file_path = os.path.join('data', 'labels', 'sample_12bit.txt') # Ensure this file exists
file_list_path = os.path.join('data', 'sample_12bit_train.txt')   # Ensure this file exists and lists the sample_image_path

# --- Crucial Checks for Test Setup ---
if not os.path.exists(sample_image_path):
    print(f"ERROR: Sample image {sample_image_path} not found. Please ensure it's created first using 'create_tiff_script.md' or similar.")
    exit()
if not os.path.exists(label_file_path):
    print(f"ERROR: Dummy label file {label_file_path} not found. Please create it with content like '0 0.5 0.5 0.5 0.5'.")
    exit()
if not os.path.exists(file_list_path):
    print(f"ERROR: File list {file_list_path} not found. Please create it and ensure it contains the relative path to the sample image (e.g., 'data/images/sample_12bit.tif').")
    exit()
# --- End Crucial Checks ---

img_size = 640 # Standard YOLOv5 image size
batch_size = 1

print(f"Attempting to load: {sample_image_path} via LoadImagesAndLabels using file list: {file_list_path}")

try:
    dataset = LoadImagesAndLabels(
        path=file_list_path,
        img_size=img_size,
        batch_size=batch_size,
        augment=True, # Test with augmentations enabled (using minimal hyp)
        hyp=hyp,
        rect=False,
        image_weights=False,
        cache_images=False, # Test direct loading path, no caching
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="Test: "
    )
    print(f"Dataset initialized. Length: {len(dataset)}")
    assert len(dataset) > 0, "Dataset could not find the image. Check file_list_path and image path."

    # Get the first (and only) item
    img_tensor, labels_out, returned_file_path, shapes = dataset[0]

    print(f"Returned file path from dataset: {returned_file_path}")
    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Image tensor dtype: {img_tensor.dtype}")
    
    tensor_min = img_tensor.min().item()
    tensor_max = img_tensor.max().item()
    print(f"Image tensor min value: {tensor_min}")
    print(f"Image tensor max value: {tensor_max}")

    # Core Assertions
    assert img_tensor.dtype == torch.float32, f"Tensor dtype is {img_tensor.dtype}, expected torch.float32"
    assert tensor_max <= 1.0 + 1e-5, f"Tensor max is {tensor_max}, expected <= 1.0 (check normalization)"
    assert tensor_min >= 0.0 - 1e-5, f"Tensor min is {tensor_min}, expected >= 0.0 (check normalization)"
    
    # Verification against raw image properties
    raw_img_cv = cv2.imread(sample_image_path, cv2.IMREAD_UNCHANGED)
    if raw_img_cv is None:
        print(f"ERROR: cv2.imread failed to load {sample_image_path} for verification. This should not happen if dataset loading succeeded.")
        exit()
        
    print(f"Raw image loaded by cv2 directly: dtype={raw_img_cv.dtype}, shape={raw_img_cv.shape}, min_val={np.min(raw_img_cv)}, max_val={np.max(raw_img_cv)}")
    
    # Simulate the dataloader's bit shift
    shifted_raw_img_cv = raw_img_cv >> 4
    print(f"Shifted raw image (simulating dataloader's initial processing): dtype={shifted_raw_img_cv.dtype}, shape={shifted_raw_img_cv.shape}, min_val={np.min(shifted_raw_img_cv)}, max_val={np.max(shifted_raw_img_cv)}")
    
    # Check if the max value of the shifted raw image is high, indicating 12-bit data was likely present
    # This assertion depends on the synthetic TIFF having values that result in >4000 after shifting.
    assert np.max(shifted_raw_img_cv) >= 4000, \
        f"Max value of shifted raw image is {np.max(shifted_raw_img_cv)}, expected to be near 4095 if synthetic TIFF was created correctly with high 12-bit values."

    print("---")
    print("TEST PASSED: Core assertions for tensor type, normalized range (0-1), and raw image high values met.")
    print("Please manually verify the printed 'Raw image loaded by cv2' and 'Shifted raw image' values to ensure the synthetic TIFF image was created and processed as expected (max value around 4095 after shift if full 12-bit range was intended in the TIFF).")
    print("---")

except Exception as e:
    print(f"An error occurred during the test: {e}")
    import traceback
    traceback.print_exc()
    print("---")
    print("TEST FAILED.")
    print("---")


print("Test script finished.")
