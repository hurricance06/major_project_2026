import os
import cv2

mask_dir = "augmented/masks/train"
count = 0

for mask_file in os.listdir(mask_dir):
    path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None or mask.size == 0:
        print(f"[REMOVED] Corrupt or empty mask: {mask_file}")
        os.remove(path)
        count += 1

print(f"✅ Cleanup complete: {count} broken masks removed.")