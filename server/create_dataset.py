import os
import pickle
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic

DATA_DIR = './data'

def normalize_keypoints(landmarks):
    """
    Normaliza los keypoints de las manos para que sean relativos al punto mínimo.
    """
    x_ = [landmark.x for landmark in landmarks]
    y_ = [landmark.y for landmark in landmarks]

    min_x, max_x = min(x_), max(x_)
    min_y, max_y = min(y_), max(y_)

    # Normaliza los keypoints para que estén en un rango de 0 a 1
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append((landmark.x - min_x) / (max_x - min_x) if max_x - min_x != 0 else 0)
        normalized_landmarks.append((landmark.y - min_y) / (max_y - min_y) if max_y - min_y != 0 else 0)

    return normalized_landmarks

def create_dataset():
    data = []
    labels = []

    with Holistic(static_image_mode=True, min_detection_confidence=0.3) as holistic:
        for dir_ in os.listdir(DATA_DIR):
            for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = holistic.process(img_rgb)

                # Verificar si hay landmarks en ambas manos
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    # Obtener los landmarks de las manos
                    landmarks = []
                    if results.left_hand_landmarks:
                        landmarks.extend(results.left_hand_landmarks.landmark)
                    if results.right_hand_landmarks:
                        landmarks.extend(results.right_hand_landmarks.landmark)

                    # Normalizar los keypoints
                    normalized_landmarks = normalize_keypoints(landmarks)
                    data.append(normalized_landmarks)
                    labels.append(dir_)

    # Guardar el dataset normalizado usando pickle
    with open('data_normalized.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print("Dataset normalizado creado y guardado en 'data_normalized.pickle'")

if __name__ == "__main__":
    create_dataset()
