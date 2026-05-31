import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("model/best.pt")

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("captures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# OPEN EXTERNAL WEBCAM
# =========================
cap = cv2.VideoCapture(1)

# =========================
# CHECK CAMERA
# =========================
if not cap.isOpened():

    print("ERROR: Webcam not opening")

    exit()

print("External webcam connected successfully")

# =========================
# MAIN LOOP
# =========================
while True:

    # =========================
    # CAPTURE FRAME
    # =========================
    ret, frame = cap.read()

    if not ret:

        print("Failed to capture frame")

        continue

    # =========================
    # TIMESTAMP
    # =========================
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # =========================
    # SAVE ORIGINAL IMAGE
    # =========================
    capture_path = (
        f"captures/{timestamp}.jpg"
    )

    cv2.imwrite(
        capture_path,
        frame
    )

    print(
        f"\nCaptured Image: "
        f"{capture_path}"
    )

    # =========================
    # RUN YOLO PREDICTION
    # =========================
    results = model(
        frame,
        conf=0.25
    )

    result = results[0]

    # =========================
    # SAVE PREDICTION IMAGE
    # =========================
    annotated_frame = result.plot()

    output_path = (
        f"outputs/{timestamp}.jpg"
    )

    cv2.imwrite(
        output_path,
        annotated_frame
    )

    print(
        f"Prediction Saved: "
        f"{output_path}"
    )

    # =========================
    # PRINT DETECTIONS
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

        found = False

        for class_id, confidence in zip(
            class_ids,
            confidences
        ):

            if confidence < 0.25:
                continue

            found = True

            class_name = model.names[
                int(class_id)
            ]

            print(
                f"Detected: "
                f"{class_name} "
                f"({confidence:.2f})"
            )

        if not found:

            print(
                "No object detected"
            )

    else:

        print("No object detected")

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow(
        "Waste Food Detection",
        annotated_frame
    )

    print(
        "\nWaiting 30 seconds..."
    )

    # =========================
    # WAIT 30 SECONDS
    # =========================
    for i in range(30):

        # press q to quit
        if cv2.waitKey(1000) & 0xFF == ord('q'):

            cap.release()

            cv2.destroyAllWindows()

            exit()

    print(
        "\nCapturing next image..."
    )

# =========================
# RELEASE CAMERA
# =========================
cap.release()

cv2.destroyAllWindows()