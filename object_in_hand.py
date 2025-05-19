from ultralytics import YOLO
import cv2
import numpy as np

class ObjectInHandDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_colors = {}
        
    def get_random_color(self):
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def detect_objects(self, frame, hand_roi):
        if hand_roi is None or hand_roi.size == 0:
            return frame, []
            
        # Run YOLO inference on the hand ROI
        results = self.model(hand_roi, conf=self.conf_threshold)
        
        detected_objects = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                # Get or create color for this class
                if class_name not in self.class_colors:
                    self.class_colors[class_name] = self.get_random_color()
                
                # Draw bounding box on the hand ROI
                cv2.rectangle(hand_roi, (x1, y1), (x2, y2), self.class_colors[class_name], 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(hand_roi, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colors[class_name], 2)
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return hand_roi, detected_objects 