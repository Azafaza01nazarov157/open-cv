import mediapipe as mp
import cv2
import numpy as np

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get pose landmarks
        results = self.pose.process(frame_rgb)
        
        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Get hand positions
            left_hand = self._get_hand_position(results.pose_landmarks, 'left')
            right_hand = self._get_hand_position(results.pose_landmarks, 'right')
            
            return frame, results.pose_landmarks, left_hand, right_hand
        
        return frame, None, None, None

    def _get_hand_position(self, landmarks, hand='right'):
        # Get the wrist landmark (15 for left wrist, 16 for right wrist)
        wrist_idx = 15 if hand == 'left' else 16
        
        if landmarks.landmark[wrist_idx]:
            wrist = landmarks.landmark[wrist_idx]
            height, width = 480, 640  # Default frame size
            
            # Convert normalized coordinates to pixel coordinates
            x = int(wrist.x * width)
            y = int(wrist.y * height)
            
            # Create a region of interest around the hand
            roi_size = 100  # Size of the ROI in pixels
            x1 = max(0, x - roi_size)
            y1 = max(0, y - roi_size)
            x2 = min(width, x + roi_size)
            y2 = min(height, y + roi_size)
            
            return (x1, y1, x2, y2)
        
        return None

    def get_hand_roi(self, frame, hand_position):
        if hand_position:
            x1, y1, x2, y2 = hand_position
            return frame[y1:y2, x1:x2]
        return None

    def __del__(self):
        self.pose.close() 