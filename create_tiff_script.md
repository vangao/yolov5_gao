```python
import cv2
import numpy as np
import os

# This script is intended to be run in an environment where OpenCV and NumPy are correctly installed.

# Ensure data/images directory exists (if running this script standalone)
# os.makedirs('data/images', exist_ok=True) # Assuming 'data/images' might not exist

print(f"Attempting to use cv2 version: {cv2.__version__}")
print(f"Attempting to use numpy version: {np.__version__}")

# Create a 10x10x3 image with uint16 data type
# The goal is to have values that, after a >> 4 shift, are close to 4095.
image_data = np.zeros((10, 10, 3), dtype=np.uint16)

# Create a gradient of 12-bit values (0-4095) then shift them to 16-bit storage
val_12bit = 1
for r in range(10):
    for c in range(10):
        for ch in range(3):
            # Store values that will be meaningful after bit shift
            # To get a value 'v' after '>> 4', store 'v << 4'
            # Add some random lower 4 bits to simulate real sensor data
            current_12bit_val = min(val_12bit, 4095) # Cap at 12-bit max
            # Ensure random part is also uint16 to prevent dtype issues during bitwise OR
            stored_val_16bit = (current_12bit_val << 4) | np.random.randint(0,16, dtype=np.uint16) 
            image_data[r, c, ch] = stored_val_16bit
            # Increment to get a range of values. 14 * 300 (10*10*3) approx 4200, covering the 12-bit range.
            val_12bit += 14 

# Specifically set a few pixels to known 12-bit values (after shift)
# Max 12-bit value (4095) stored with 0xF in lower bits (becomes 65535 in uint16)
image_data[0, 0, 0] = (4095 << 4) | 0b1111 
# A high 12-bit value
image_data[0, 1, 0] = (4000 << 4) | 0b1010 
# A mid 12-bit value
image_data[0, 2, 0] = (2048 << 4) | 0b0101 
# A low 12-bit value
image_data[1, 0, 0] = (100 << 4)  | 0b0011 
# Zero 12-bit value (black)
image_data[1, 1, 0] = (0 << 4)    | 0b0000

print(f"Max value in image_data before saving: {np.max(image_data)}")

# Define output path, assuming 'data/images/' directory structure as used in YOLOv5
output_dir = 'data/images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'sample_12bit.tif')

try:
    success = cv2.imwrite(output_path, image_data)
    if success:
        print(f"Synthetic 16-bit TIFF image saved to {output_path}")
    else:
        print(f"ERROR: cv2.imwrite failed to save to {output_path}. Check directory permissions and OpenCV setup.")
except Exception as e:
    print(f"ERROR during cv2.imwrite: {e}")
    print("Ensure that the directory structure (e.g., data/images/) exists or can be created by this script if it has permissions.")

# Verification read (optional, but good for testing the script itself)
if os.path.exists(output_path) and success:
    try:
        img_read = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        if img_read is not None:
            print(f"TIFF Verification (after saving): dtype={img_read.dtype}, shape={img_read.shape}, min_val={np.min(img_read)}, max_val={np.max(img_read)}")
        else:
            print(f"ERROR: Could not read back {output_path} for verification using cv2.imread.")
    except Exception as e:
        print(f"ERROR during verification read: {e}")
else:
    print(f"Skipping verification read as image was not saved successfully or does not exist.")

print("Script finished.")
```
