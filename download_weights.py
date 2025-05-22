import torch
import os

try:
    print("Attempting to download yolov5s.pt using torch.hub.load...")
    # This will load the model and it typically saves the .pt file to a cache directory
    # or sometimes to the current working directory if specific ultralytics versions behave that way.
    # Forcing a specific path for the .pt file with torch.hub.load is not straightforward,
    # as it depends on the repository's hubconf.py.
    # However, the standard behavior of ultralytics/yolov5 hubconf should make it available.
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("Model loaded successfully.")

    # The .pt file is usually saved in torch.hub.get_dir() + '/ultralytics_yolov5_master' or similar
    # or potentially in ~/.cache/torch/hub/ultralytics_yolov5_master/
    # Let's try to locate it and copy it to the current directory if needed.
    # A simpler way for yolov5 is that it often creates/checks for 'yolov5s.pt' in the working dir
    # or uses the one from its cache.
    
    # The `attempt_download` function in `utils/downloads.py` from yolov5 *should* do this more directly.
    # Since that failed, this is an alternative.
    # After loading, yolov5s.pt should be in the local directory or accessible.
    # Let's check if 'yolov5s.pt' got created in current dir by the above command.
    if os.path.exists('yolov5s.pt'):
        print("yolov5s.pt found in the current directory after torch.hub.load.")
    else:
        # If not in current directory, it might be in the torch hub cache.
        # Ultralytics' `attempt_download` usually handles this.
        # This is a fallback, let's see if the detect script can find it via its internal mechanisms
        # now that it's been cached by torch.hub.
        print("yolov5s.pt was not placed in the current directory by torch.hub.load.")
        print("It should be in the torch hub cache. Trying to run detect.py which might find it.")
        # As a more robust measure, let's try the utils.downloads again, as it might behave differently
        # now that the files are definitely in the cache.
        from utils.downloads import attempt_download
        print("Attempting download via utils.downloads.attempt_download again...")
        attempt_download('yolov5s.pt')
        if os.path.exists('yolov5s.pt'):
            print("yolov5s.pt is now present in the current directory after attempt_download.")
        else:
            print("Failed to make yolov5s.pt available in current directory even after attempt_download.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("If the error is 'No module named 'utils'', ensure you are running from the yolov5 root directory.")

print("Script finished.")
