import os
import json
from pathlib import Path

def coco_seg_to_yolov8(coco_path, image_dir, label_dir, category_remap):
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    # Case-insensitive category mapping
    categories = {}
    skipped = []
    for cat in coco['categories']:
        name = cat['name'].strip().lower()
        matched = False
        for key in category_remap:
            if key.strip().lower() == name:
                categories[cat['id']] = category_remap[key]
                matched = True
                break
        if not matched:
            skipped.append((cat['id'], cat['name']))

    if skipped:
        print("⚠️ Skipped categories (not in category_remap):")
        for cid, cname in skipped:
            print(f"  - ID {cid}: '{cname}'")

    # Ensure label dir exists
    os.makedirs(label_dir, exist_ok=True)

    # Group annotations by image
    anns_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if ann['category_id'] not in categories:
            continue
        anns_by_image.setdefault(img_id, []).append(ann)

    # Create YOLOv8-style label files
    for img in coco['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_w, img_h = img['width'], img['height']

        label_file = Path(label_dir) / Path(img_name).with_suffix('.txt')
        with open(label_file, 'w') as f:
            for ann in anns_by_image.get(img_id, []):
                cat_id = ann['category_id']
                seg = ann.get('segmentation', [])

                if not seg or not isinstance(seg[0], list):
                    continue  # skip RLE or invalid

                # Normalize segmentation points
                normalized_seg = []
                for i in range(0, len(seg[0]), 2):
                    x = seg[0][i] / img_w
                    y = seg[0][i+1] / img_h
                    normalized_seg.extend([x, y])

                flat_str = ' '.join([f"{p:.6f}" for p in normalized_seg])
                f.write(f"{categories[cat_id]} {flat_str}\n")

    print(f"✅ Conversion completed: {label_dir}")



if __name__ == "__main__":
    base = "C:/Users/abhin/OneDrive/Documents/GitHub/Albumentationsx/Minor_Project_Dataset.v9i.coco-segmentation"

    # Map your custom class names exactly as per data.yaml
    category_remap = {
        "Rice": 0,
        "Roti": 1,
        "aloo-bhurji": 2,
        "daal-pulse": 3,
        "Besan-Chilla": 4,
        "Biryani": 5,
        "Bundi": 6,
        "Channa-Daal": 7,
        "Chilli-paneer": 8,
        "Chutney": 9,
        "Daal-pulse": 10,
        "gulab-jamun": 11,
        "halwa": 12,
        "jalebi": 13,
        "Mushroom": 14,
        "No-waste": 15,
        "Palak-paneer": 16,
        "Paneer-bhurji": 17,
        "Papad": 18,
        "Raita": 19,
        "Rajma": 20,
        "Red--chutney": 21,
        "Rice-with-pulse": 22,
        "Salad-Chukundar": 23,
        "Salad-Mix": 24,
        "Salad-kheera": 25,
        "Salad-onion": 26,
        "aloo-shimla": 27,
        "gajar-aloo": 28,
        "kadi-pakoda": 29,
        "papad": 30,
        "pudi": 31,
        "tea": 32
    }

    # Paths to annotations
    coco_train = os.path.join(base, "train/_annotations.coco.json")
    coco_val = os.path.join(base, "valid/_annotations.coco.json")

    # Convert
    print("✅ Starting conversion...")
    coco_seg_to_yolov8(coco_train, f"{base}/train/images", f"{base}/train/labels", category_remap)
    coco_seg_to_yolov8(coco_val, f"{base}/valid/images", f"{base}/valid/labels", category_remap)