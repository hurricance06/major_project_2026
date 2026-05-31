# Waste Food Detection

## Overview

This project is a major project for waste food detection using a YOLO object detection model. It captures images from an external webcam, runs predictions using a trained model, saves annotated output images, and generates CSV reports for detected food items.

## Project Structure

- `main.py` - Captures frames from an external webcam, runs YOLO predictions, saves raw captures to `captures/`, saves annotated predictions to `outputs/`, and displays results.
- `generate_report.py` - Processes new images from `captures/`, runs YOLO detection, counts detected food items, and saves summary reports to `reports/`.
- `testcam.py` - A simple webcam test script to verify webcam connection and live feed.
- `model/best.pt` - The trained YOLO model weights used for detection.
- `captures/` - Folder for raw captured images.
- `outputs/` - Folder for prediction output images.
- `reports/` - Folder for generated CSV reports.
- `processed.txt` - Tracks which images have already been processed, so reports are generated only for new images.

## Requirements

- Python 3.9+ recommended
- `opencv-python`
- `pandas`
- `ultralytics`

Install dependencies with:

```bash
pip install opencv-python pandas ultralytics
```

## Usage

1. Place the trained YOLO model in `model/best.pt`.
2. Connect an external webcam to your computer.
3. Run the main capture and detection script:

```bash
python main.py
```

4. After capturing images, generate a detection report:

```bash
python generate_report.py
```

5. To verify webcam connectivity before running the main script, use:

```bash
python testcam.py
```

## Output

- Captured images are saved in `captures/`
- Annotated detection images are saved in `outputs/`
- Reports are saved in `reports/` as CSV files like `report_001.csv`

## Notes

- The capture script waits 30 seconds between images and allows quitting with `q` while the display window is active.
- `generate_report.py` processes up to 10 new images per run and ignores images already listed in `processed.txt`.
- Adjust detection confidence thresholds inside the scripts if you need more or fewer detections.

## Future Improvements

- Add a configuration file for webcam index, confidence thresholds, and batch size.
- Improve report details with timestamps, image names, and class probability.
- Add a GUI or dashboard for monitoring detections and reports.
