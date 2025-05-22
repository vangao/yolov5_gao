import cv2
import numpy as np
import torch
import os
import yaml 

print(f"cv2 version for test_12bit_loader.py: {cv2.__version__}")
print(f"numpy version for test_12bit_loader.py: {np.__version__}")
print(f"torch version for test_12bit_loader.py: {torch.__version__}")

import sys
# Corrected sys.path to point to the root for utils.dataloaders
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 

from utils.dataloaders import LoadImagesAndLabels

print("Starting 12-bit loader test script...")

hyp = {
    'mosaic': 0.0, 'mixup': 0.0, 'degrees': 0.0, 'translate': 0.0, 
    'scale': 0.0, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 
    'fliplr': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 
    'copy_paste': 0.0
}

sample_image_filename = 'sample_12bit.tif'
sample_image_path = os.path.join('data', 'images', sample_image_filename)
label_file_path = os.path.join('data', 'labels', 'sample_12bit.txt')
file_list_path = os.path.join('data', 'sample_12bit_train.txt')

if not os.path.exists(sample_image_path):
    print(f"ERROR: Sample image {sample_image_path} not found. Please ensure it's created.")
    exit()
if not os.path.exists(label_file_path):
    print(f"ERROR: Dummy label file {label_file_path} not found.")
    exit()
if not os.path.exists(file_list_path):
    print(f"ERROR: File list {file_list_path} not found.")
    exit()

img_size = 640
batch_size = 1

print(f"Attempting to load: {sample_image_path} via LoadImagesAndLabels")

try:
    dataset = LoadImagesAndLabels(
        path=file_list_path,
        img_size=img_size,
        batch_size=batch_size,
        augment=True, 
        hyp=hyp,
        rect=False,
        image_weights=False,
        cache_images=False, 
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="Test: "
    )
    print(f"Dataset initialized. Length: {len(dataset)}")
    assert len(dataset) > 0, "Dataset could not find the image."

    img_tensor, labels_out, returned_file_path, shapes = dataset[0]

    print(f"Returned file path: {returned_file_path}")
    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Image tensor dtype: {img_tensor.dtype}")
    
    tensor_min = img_tensor.min().item()
    tensor_max = img_tensor.max().item()
    print(f"Image tensor min value: {tensor_min}")
    print(f"Image tensor max value: {tensor_max}")

    assert img_tensor.dtype == torch.float32, f"Tensor dtype is {img_tensor.dtype}, expected torch.float32"
    assert tensor_max <= 1.0 + 1e-5, f"Tensor max is {tensor_max}, expected <= 1.0"
    assert tensor_min >= 0.0 - 1e-5, f"Tensor min is {tensor_min}, expected >= 0.0"
    
    raw_img_cv = cv2.imread(sample_image_path, cv2.IMREAD_UNCHANGED)
    print(f"Raw image loaded by cv2 directly: dtype={raw_img_cv.dtype}, shape={raw_img_cv.shape}, min={np.min(raw_img_cv)}, max={np.max(raw_img_cv)}")
    
    shifted_raw_img_cv = raw_img_cv >> 4
    print(f"Shifted raw image (if done manually): dtype={shifted_raw_img_cv.dtype}, shape={shifted_raw_img_cv.shape}, min={np.min(shifted_raw_img_cv)}, max={np.max(shifted_raw_img_cv)}")
    
    assert np.max(shifted_raw_img_cv) >= 4000, f"Max value of shifted raw image is {np.max(shifted_raw_img_cv)}, expected to be near 4095."

    print("Test assertions passed for tensor type, normalized range (0-1), and raw image high values.")

except Exception as e:
    print(f"An error occurred during the test: {e}")
    import traceback
    traceback.print_exc()

print("Test script finished.")
