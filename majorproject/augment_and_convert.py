# augment_and_convert.py
import os
import cv2
import numpy as np
from albumentations import (
    Compose, RandomRotate90, Rotate, HueSaturationValue,
    RandomBrightnessContrast, MotionBlur, HorizontalFlip
)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Set paths
IMG_DIR = "dataset/images/train"
MASK_DIR = "dataset/masks/train"
AUG_IMG_DIR = "augmented/images/train"
AUG_MASK_DIR = "augmented/masks/train"

os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

# Augmentations
transform = Compose([
    Rotate(limit=15, p=0.7),
    HueSaturationValue(p=0.7),
    RandomBrightnessContrast(p=0.7),
    MotionBlur(blur_limit=3, p=0.5),
    HorizontalFlip(p=0.5)
])
# Additional augmentations
transform = Compose([
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    cv2.ColorJitter(p=0.3),
    cv2.GaussianBlur(p=0.2),
    RandomBrightnessContrast(p=0.4),
    cv2.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
])

# Process images
for img_name in tqdm(os.listdir(IMG_DIR)):
    image = cv2.imread(os.path.join(IMG_DIR, img_name))
    mask = cv2.imread(os.path.join(MASK_DIR, img_name), cv2.IMREAD_GRAYSCALE)

    for i in range(3):  # 3 augmentations per image
        augmented = transform(image=image, mask=mask)
        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        base_name = f"{os.path.splitext(img_name)[0]}_aug{i}.png"
        cv2.imwrite(os.path.join(AUG_IMG_DIR, base_name), aug_img)
        cv2.imwrite(os.path.join(AUG_MASK_DIR, base_name), aug_mask)