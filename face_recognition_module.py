import cv2
import numpy as np
import logging
import os

class FaceRecognizer:
    def __init__(self):
        # Загрузка предобученных моделей OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Создаем LBPH распознаватель лиц
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Словарь для хранения известных лиц
        self.known_faces = {}
        self.face_id = 0
        
        # Создаем директорию для хранения лиц, если её нет
        if not os.path.exists('faces'):
            os.makedirs('faces')
    
    def detect_faces(self, frame):
        # Конвертация в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекция лиц
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_data = []
        for (x, y, w, h) in faces:
            # Получаем ROI лица
            face_roi = gray[y:y+h, x:x+w]
            
            # Детекция глаз
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            # Если найдены глаза, считаем что это реальное лицо
            if len(eyes) > 0:
                face_data.append({
                    'bbox': (x, y, w, h),
                    'roi': face_roi
                })
                
                # Отрисовка рамки
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Пытаемся распознать лицо
                try:
                    face_id, confidence = self.face_recognizer.predict(face_roi)
                    if confidence < 50:  # Порог уверенности
                        name = self.known_faces.get(face_id, "Unknown")
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except:
                    cv2.putText(frame, "Unknown", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, face_data
    
    def add_known_face(self, name, face_image):
        """Добавление нового известного лица в базу"""
        try:
            # Конвертация в градации серого
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Детекция лица
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Сохраняем лицо
                face_id = len(self.known_faces)
                self.known_faces[face_id] = name
                
                # Сохраняем изображение
                cv2.imwrite(f'faces/{name}_{face_id}.jpg', face_roi)
                
                # Обучаем распознаватель
                self.face_recognizer.train([face_roi], np.array([face_id]))
                
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error adding known face: {str(e)}")
            return False
    
    def load_known_faces(self):
        """Загрузка известных лиц из директории"""
        try:
            face_images = []
            face_ids = []
            
            for filename in os.listdir('faces'):
                if filename.endswith('.jpg'):
                    # Извлекаем ID из имени файла
                    face_id = int(filename.split('_')[1].split('.')[0])
                    name = filename.split('_')[0]
                    
                    # Загружаем изображение
                    face_image = cv2.imread(os.path.join('faces', filename), cv2.IMREAD_GRAYSCALE)
                    
                    face_images.append(face_image)
                    face_ids.append(face_id)
                    self.known_faces[face_id] = name
            
            if face_images:
                self.face_recognizer.train(face_images, np.array(face_ids))
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error loading known faces: {str(e)}")
            return False 