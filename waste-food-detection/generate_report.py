import os
import pandas as pd
from ultralytics import YOLO

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("model/best.pt")

# =========================
# FOLDERS
# =========================
output_folder = "captures"
report_folder = "reports"
processed_file = "processed.txt"

os.makedirs(report_folder, exist_ok=True)

# =========================
# LOAD PROCESSED IMAGES
# =========================
processed_images = set()

if os.path.exists(processed_file):

    with open(processed_file, "r") as f:

        processed_images = set(
            line.strip() for line in f
        )

# =========================
# GET ALL IMAGES
# =========================
all_images = [
    img for img in os.listdir(output_folder)
    if img.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

# =========================
# REMOVE PROCESSED IMAGES
# =========================
new_images = [
    img for img in all_images
    if img not in processed_images
]

# =========================
# TAKE NEXT 10 IMAGES
# =========================
batch_images = new_images[:10]

if len(batch_images) == 0:

    print("No new images found")

    exit()

# =========================
# FOOD COUNTS
# =========================
food_counts = {}

# =========================
# PROCESS IMAGES
# =========================
for image_name in batch_images:

    image_path = os.path.join(
        output_folder,
        image_name
    )

    print(f"\nProcessing: {image_name}")

    # =========================
    # RUN YOLO PREDICTION
    # =========================
    results = model(
        image_path,
        conf=0.01
    )

    result = results[0]

    # =========================
    # CHECK DETECTIONS
    # =========================
    if result.boxes is not None:

        class_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        # =========================
        # STORE BEST CONFIDENCE
        # =========================
        best_detections = {}

        for class_id, confidence in zip(
            class_ids,
            confidences
        ):

            class_name = model.names[
                int(class_id)
            ]

            # Keep highest confidence only
            if (
                class_name not in best_detections
                or confidence >
                best_detections[class_name]
            ):

                best_detections[
                    class_name
                ] = confidence

        # =========================
        # FINAL COUNTING
        # =========================
        for class_name, confidence in (
            best_detections.items()
        ):

            print(
                f"{class_name} confidence = "
                f"{confidence:.2f}"
            )

            # Ignore weak predictions
            if confidence < 0.10:
                continue

            print(
                f"Detected: {class_name} "
                f"({confidence:.2f})"
            )

            # Count each class once per image
            if class_name in food_counts:

                food_counts[class_name] += 1

            else:

                food_counts[class_name] = 1

# =========================
# CHECK IF ANY DETECTIONS
# =========================
if len(food_counts) == 0:

    print(
        "\nNo food items detected "
        "above confidence threshold"
    )

    exit()

# =========================
# CREATE REPORT TABLE
# =========================
report_df = pd.DataFrame(
    list(food_counts.items()),
    columns=[
        "Food Item",
        "Count"
    ]
)

# =========================
# SAVE REPORT
# =========================
report_number = (
    len(os.listdir(report_folder)) + 1
)

report_path = (
    f"{report_folder}/"
    f"report_{report_number:03d}.csv"
)

report_df.to_csv(
    report_path,
    index=False
)

# =========================
# MARK IMAGES AS PROCESSED
# =========================
with open(processed_file, "a") as f:

    for image in batch_images:

        f.write(f"{image}\n")

# =========================
# SHOW REPORT
# =========================
print("\nREPORT GENERATED\n")

print(report_df)

print(f"\nSaved: {report_path}")