import cv2
import numpy as np

# Define the image path
image_path = 'data/images/2-2.tiff'

# Load the image
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image loaded successfully
if img is None:
    print(f"Error: Could not load image at {image_path}")
else:
    # Print image properties
    print(f"Image dtype: {img.dtype}")
    print(f"Image shape: {img.shape}")
    print(f"Minimum pixel value: {np.min(img)}")
    print(f"Maximum pixel value: {np.max(img)}")
