import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

# ----------------- SETTINGS -----------------
dataset_path = "Major_Project_Dataset.v9i.coco-segmentation/dataset2"

img_dir = os.path.join(dataset_path, "valid/images")
label_dir = os.path.join(dataset_path, "valid/labels")

# ----------------- LOAD CLASS NAMES (FIXED) -----------------

def load_classes():
    json_path = os.path.join(dataset_path, "_annotations.coco.json")

    if not os.path.exists(json_path):
        print("❌ JSON not found:", json_path)
        return []

    with open(json_path, "r") as f:
        coco = json.load(f)

    # SORT BY ID (VERY IMPORTANT)
    categories = sorted(coco["categories"], key=lambda x: x["id"])

    class_names = [cat["name"] for cat in categories]

    return class_names


class_names = load_classes()

print("✅ Loaded classes:", len(class_names))

# ----------------- DRAW FUNCTION -----------------

def draw_segmentation(image, label_path):
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            cls = int(parts[0])
            segmentation = list(map(float, parts[1:]))

            points = np.array(segmentation).reshape(-1, 2)
            points *= np.array([w, h])
            points = points.astype(np.int32)

            cv2.polylines(image, [points], True, (0,255,0), 2)

            # SAFE NAME (NO CRASH)
            if cls < len(class_names):
                name = class_names[cls]
            else:
                name = f"class_{cls}"

            cv2.putText(image, name, tuple(points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    return image

# ----------------- VISUALIZE -----------------

image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))[:5]

for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    annotated = draw_segmentation(image.copy(), label_path)

    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(base)
    plt.axis('off')
    plt.show()