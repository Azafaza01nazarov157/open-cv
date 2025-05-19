# Real-time Object Detection System

This project implements a real-time object detection system using YOLOv8 and OpenCV. It can detect various objects in video streams from webcams or video files.

## Features

- Real-time object detection using YOLOv8
- Support for webcam and video file input
- Bounding box visualization with class labels and confidence scores
- Automatic video recording of detection results
- Detailed logging of detected objects
- Configurable confidence threshold

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the detection script:
```bash
python detect.py
```

2. The program will:
   - Open your webcam (or use a video file if specified)
   - Display the video feed with detected objects
   - Save the processed video to `video_output/output.avi`
   - Log detections to `logs/detection.log`

3. Press 'q' to quit the program

## Configuration

You can modify the following parameters in `detect.py`:

- `model_path`: Path to the YOLO model (default: 'yolov8n.pt')
- `conf_threshold`: Minimum confidence threshold for detections (default: 0.25)
- Video source: Change `cv2.VideoCapture(0)` to use a different camera or video file

## Project Structure

```
object-detection-rt/
├── detect.py                # Main detection script
├── video_output/            # Output videos
├── logs/                    # Detection logs
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- NumPy
- Other dependencies listed in requirements.txt

## Notes

- The program uses YOLOv8n by default, which is a lightweight model suitable for real-time detection
- For better accuracy, you can use larger YOLOv8 models (s, m, l, x)
- Processing speed depends on your hardware and the selected model 