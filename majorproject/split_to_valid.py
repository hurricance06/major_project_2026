import os
import shutil

train_img = "Major_Project_Dataset.v9i.coco-segmentation/dataset2/train/images"
train_lbl = "Major_Project_Dataset.v9i.coco-segmentation/dataset2/train/labels"

val_img = "Major_Project_Dataset.v9i.coco-segmentation/dataset2/valid/images"
val_lbl = "Major_Project_Dataset.v9i.coco-segmentation/dataset2/valid/labels"

os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = sorted(os.listdir(train_img))

moved = 0

for i, img in enumerate(images):
    if i % 5 == 0:  # every 5th image → validation
        label = img.replace(".jpg", ".txt")

        src_img = os.path.join(train_img, img)
        src_lbl = os.path.join(train_lbl, label)

        dst_img = os.path.join(val_img, img)
        dst_lbl = os.path.join(val_lbl, label)

        if os.path.exists(src_lbl):
            shutil.move(src_img, dst_img)
            shutil.move(src_lbl, dst_lbl)
            moved += 1

print(f"✅ Moved {moved} images to validation")