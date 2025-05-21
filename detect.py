import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import logging
from datetime import datetime
import os
from pose_utils import PoseDetector
from object_in_hand import ObjectInHandDetector
from tracker import Sort
from face_recognition_module import FaceRecognizer
import pandas as pd

# Create necessary directories first
os.makedirs('logs', exist_ok=True)
os.makedirs('video_output', exist_ok=True)
os.makedirs('analytics', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/detection.log'),
        logging.StreamHandler()
    ]
)

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_colors = {}
        self.pose_detector = PoseDetector()
        self.object_in_hand_detector = ObjectInHandDetector()
        self.tracker = Sort()
        self.face_recognizer = FaceRecognizer()
        
    def get_random_color(self):
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def process_frame(self, frame):
        # Process face recognition
        frame, face_data = self.face_recognizer.detect_faces(frame)
        
        # Process pose detection
        frame, pose_landmarks, left_hand, right_hand = self.pose_detector.process_frame(frame)
        
        # Process objects in hands
        if left_hand:
            hand_roi = self.pose_detector.get_hand_roi(frame, left_hand)
            if hand_roi is not None:
                hand_roi, left_objects = self.object_in_hand_detector.detect_objects(frame, hand_roi)
                if left_objects:
                    logging.info(f"Left hand objects: {[obj['class'] for obj in left_objects]}")
        
        if right_hand:
            hand_roi = self.pose_detector.get_hand_roi(frame, right_hand)
            if hand_roi is not None:
                hand_roi, right_objects = self.object_in_hand_detector.detect_objects(frame, hand_roi)
                if right_objects:
                    logging.info(f"Right hand objects: {[obj['class'] for obj in right_objects]}")
        
        # Run YOLOv8 inference for general object detection
        results = self.model(frame, conf=self.conf_threshold)
        
        # Prepare detections for tracking
        detections = []
        class_names = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                # Add to detections
                detections.append([x1, y1, x2, y2, confidence])
                class_names.append(class_name)
        
        # Update tracker
        if len(detections) > 0:
            detections = np.array(detections)
            tracked_objects = self.tracker.update(dets=detections, class_names=class_names)
            
            # Draw tracked objects
            for obj in tracked_objects:
                try:
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, obj[:4])
                    track_id = int(obj[4])
                    class_name = str(obj[5])
                    
                    # Get or create color for this track
                    if track_id not in self.class_colors:
                        self.class_colors[track_id] = self.get_random_color()
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_colors[track_id], 2)
                    
                    # Draw label with track ID
                    label = f"{class_name} ID:{track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colors[track_id], 2)
                except (ValueError, IndexError) as e:
                    logging.error(f"Error processing tracked object: {e}")
                    continue
        
        return frame

    def save_analytics(self):
        # Save tracking analytics to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analytics_file = f'analytics/tracking_{timestamp}.csv'
        self.tracker.analytics.to_csv(analytics_file, index=False)
        logging.info(f"Analytics saved to {analytics_file}")

def main():
    # Initialize detector
    detector = ObjectDetector()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Check if camera opened successfully
    if not cap.isOpened():
        logging.error("Error: Could not open camera")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_output/output.avi', fourcc, fps, (width, height))
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Object Detection', processed_frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
    
    finally:
        # Save analytics
        detector.save_analytics()
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated successfully")

if __name__ == "__main__":
    main() 