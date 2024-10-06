import os
import pickle
import numpy as np
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS

from tts import text_to_speech

# Cargar el modelo entrenado y el mapeo de etiquetas
model_data = pickle.load(open('model.p', 'rb'))
model = model_data['model']
model_type = model_data['model_type']
label_map = model_data['label_map']

# Parámetros para secuencia
max_sequence_length = 30  # Longitud fija de la secuencia
num_features = 84  # Número de características: 21 puntos * 2 coordenadas * 2 manos

# Función para normalizar los keypoints
def normalize_keypoints(landmarks):
    x_ = [landmark.x for landmark in landmarks]
    y_ = [landmark.y for landmark in landmarks]

    min_x, max_x = min(x_), max(x_)
    min_y, max_y = min(y_), max(y_)

    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append((landmark.x - min_x) / (max_x - min_x) if max_x - min_x != 0 else 0)
        normalized_landmarks.append((landmark.y - min_y) / (max_y - min_y) if max_y - min_y != 0 else 0)

    return normalized_landmarks

# Función para extraer keypoints de ambas manos
def extract_keypoints(results):
    # Obtener landmarks de ambas manos
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else [np.zeros(21 * 2)]
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else [np.zeros(21 * 2)]

    # Normalizar puntos clave si las manos están presentes
    left_hand_keypoints = normalize_keypoints(left_hand_landmarks) if results.left_hand_landmarks else [0] * 42
    right_hand_keypoints = normalize_keypoints(right_hand_landmarks) if results.right_hand_landmarks else [0] * 42

    # Combinar puntos clave de ambas manos
    return left_hand_keypoints + right_hand_keypoints

# Función para hacer predicciones con secuencias
def predict_sequence(model, sequence):
    # Convertir la secuencia a un arreglo NumPy con la forma adecuada
    sequence = np.array(sequence)
    
    # Ajustar entrada según el tipo de modelo
    if model_type == 'LSTM':
        sequence = sequence.reshape(1, max_sequence_length, num_features)
        prediction = model.predict(sequence)
        prediction = np.argmax(prediction, axis=1)
    elif model_type == 'Random Forest':
        sequence = sequence.flatten().reshape(1, -1)  # Aplanar la secuencia
        prediction = model.predict(sequence)
    
    return label_map[int(prediction[0])]

# Inicializar MediaPipe Holistic
with Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para salir")

    # Lista para almacenar la secuencia de keypoints
    sequence = []
    last_prediction = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar imagen
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Dibujar los puntos clave de ambas manos
        if results.left_hand_landmarks:
            draw_landmarks(frame, results.left_hand_landmarks, HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            draw_landmarks(frame, results.right_hand_landmarks, HAND_CONNECTIONS)
        
        # Extraer los keypoints de ambas manos
        keypoints = extract_keypoints(results)

        # Agregar keypoints a la secuencia si tienen la longitud correcta
        if len(keypoints) == num_features:
            sequence.append(keypoints)

                text_to_speech(prediction)
                print(f"La palabra es {prediction}")

                # Mostrar un cuadro de texto semitransparente en la parte inferior
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Controlar el tamaño de la secuencia
        if len(sequence) > max_sequence_length:
            sequence.pop(0)  # Mantener solo los últimos 'max_sequence_length' frames

        # Hacer predicción si la secuencia está completa
        if len(sequence) == max_sequence_length:
            prediction = predict_sequence(model, sequence)

            # Mostrar la palabra detectada solo si es diferente de la anterior
            if prediction != last_prediction:
                last_prediction = prediction

            # Mostrar un cuadro de texto semitransparente en la parte inferior
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Mostrar la predicción en la ventana
            cv2.putText(frame, f'Detectado: {last_prediction}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar la ventana con la cámara
        cv2.imshow('Sign Language Detection', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
