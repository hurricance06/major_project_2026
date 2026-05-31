import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- Dataset ---
class FoodSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, num_classes=4):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"[WARNING] Missing or unreadable file: {image_name}")
            return self.__getitem__((idx + 1) % len(self.image_list))

        if image.shape[:2] != mask.shape[:2]:
            print(f"[ERROR] Shape mismatch in {image_name}: {image.shape[:2]} vs {mask.shape[:2]}")
            return self.__getitem__((idx + 1) % len(self.image_list))

    # Fix invalid values
        if mask.max() >= self.num_classes:
            print(f"[WARNING] High class index ({mask.max()}) in: {image_name} -> clipped")
            mask[mask == 255] = 0
            mask = np.clip(mask, 0, self.num_classes - 1)

        augmented = self.transform(image=image, mask=mask)
        if augmented["image"] is None or augmented["mask"] is None:
            print(f"[ERROR] Augmentation failed for: {image_name}")
            return self.__getitem__((idx + 1) % len(self.image_list))

        return augmented["image"], augmented["mask"].long()


# --- Paths ---
train_img_path = "augmented/images/train"
train_mask_path = "augmented/masks/train"

# --- Training Parameters ---
NUM_CLASSES = 4  # e.g., 0: background, 1: rice, 2: curry, 3: curd/salad
BATCH_SIZE = 4
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform ---
train_transform = Compose([
    Resize(256, 256),
    Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
], is_check_shapes=False)

# --- Dataset & Loader ---
train_dataset = FoodSegDataset(train_img_path, train_mask_path, transform=train_transform, num_classes=NUM_CLASSES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = smp.Unet(
    encoder_name="resnet50",        # or "efficientnet-b0"
    encoder_weights="imagenet",     # pretrained encoder
    in_channels=3,
    classes=NUM_CLASSES
)
model.to(DEVICE)

# --- Loss & Optimizer ---
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[INFO] Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss / len(train_loader):.4f}")