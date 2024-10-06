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

def process_image(img_path, holistic):
    """
    Procesa una imagen y devuelve los keypoints normalizados si se detectan las manos.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    # Obtener landmarks de ambas manos
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else [np.zeros(21 * 2)]
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else [np.zeros(21 * 2)]

    # Normalizar puntos clave si las manos están presentes
    left_hand_keypoints = normalize_keypoints(left_hand_landmarks) if results.left_hand_landmarks else [0] * 42
    right_hand_keypoints = normalize_keypoints(right_hand_landmarks) if results.right_hand_landmarks else [0] * 42

    # Combinar puntos clave de ambas manos
    combined_keypoints = left_hand_keypoints + right_hand_keypoints

    # Verificar que siempre se obtienen el mismo número de características (num_features)
    if len(combined_keypoints) != 84:  # 21 puntos por mano, 2 coordenadas cada uno, 2 manos
        return None  # Ignorar si no tiene la forma correcta
    return combined_keypoints

def create_dataset():
    data = []
    labels = []
    max_sequence_length = 30  # Definir una longitud fija para todas las secuencias
    num_features = 84  # Número de características por keypoint (21 puntos * 2 coordenadas * 2 manos)

    with Holistic(static_image_mode=True, min_detection_confidence=0.3) as holistic:
        for dir_ in os.listdir(DATA_DIR):
            dir_path = os.path.join(DATA_DIR, dir_)
            if os.path.isdir(dir_path):
                # Verificar si es una palabra dinámica (contiene subcarpetas con secuencias)
                if any(os.path.isdir(os.path.join(dir_path, sub)) for sub in os.listdir(dir_path)):
                    # Procesar las secuencias para las palabras dinámicas
                    for sequence_folder in os.listdir(dir_path):
                        sequence_path = os.path.join(dir_path, sequence_folder)
                        if os.path.isdir(sequence_path):
                            sequence_data = []
                            for img_name in sorted(os.listdir(sequence_path)):
                                img_path = os.path.join(sequence_path, img_name)
                                keypoints = process_image(img_path, holistic)
                                if keypoints:
                                    sequence_data.append(keypoints)
                            
                            # Rellenar o recortar la secuencia para que tenga la longitud fija
                            if len(sequence_data) < max_sequence_length:
                                padding = [np.zeros(num_features) for _ in range(max_sequence_length - len(sequence_data))]
                                sequence_data.extend(padding)
                            elif len(sequence_data) > max_sequence_length:
                                sequence_data = sequence_data[:max_sequence_length]
                            
                            # Verificar que todas las secuencias tienen el tamaño correcto antes de añadir
                            if len(sequence_data) == max_sequence_length and all(len(frame) == num_features for frame in sequence_data):
                                data.append(sequence_data)
                                labels.append(dir_)
                else:
                    # Procesar imágenes individuales para palabras estáticas
                    for img_name in os.listdir(dir_path):
                        img_path = os.path.join(dir_path, img_name)
                        keypoints = process_image(img_path, holistic)
                        if keypoints:
                            # Crear una secuencia repetida para las palabras estáticas
                            seq_padded = [keypoints for _ in range(max_sequence_length)]
                            # Verificar que la secuencia tiene el tamaño correcto
                            if all(len(frame) == num_features for frame in seq_padded):
                                data.append(seq_padded)
                                labels.append(dir_)

    # Convertir los datos en un arreglo de NumPy
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    # Guardar el dataset normalizado usando pickle
    with open('data_normalized.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print("Dataset normalizado creado y guardado en 'data_normalized.pickle'")

if __name__ == "__main__":
    create_dataset()
