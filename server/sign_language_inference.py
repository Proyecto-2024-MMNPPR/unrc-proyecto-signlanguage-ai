import os
import pickle
import numpy as np
import cv2
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks

# Cargar el modelo entrenado y el mapeo de etiquetas
model_data = pickle.load(open('model.p', 'rb'))
model = model_data['model']
model_type = model_data['model_type']
label_map = model_data['label_map']

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

# Función para hacer predicciones
def predict(model, keypoints):
    keypoints = np.array(keypoints).reshape(1, -1)

    # Si el modelo es LSTM, ajustar las dimensiones
    if model_type == 'LSTM':
        keypoints = np.expand_dims(keypoints, axis=1)
    
    prediction = model.predict(keypoints)
    
    if model_type == 'LSTM':
        prediction = np.argmax(prediction, axis=1)
    
    return label_map[int(prediction[0])]

# Inicializar MediaPipe Holistic
with Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para salir")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar imagen
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Detección de una mano (izquierda o derecha)
        landmarks = []
        if results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark
            draw_landmarks(frame, results.left_hand_landmarks, HAND_CONNECTIONS)
        elif results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            draw_landmarks(frame, results.right_hand_landmarks, HAND_CONNECTIONS)
        
        if landmarks:
            # Normalizar keypoints
            keypoints = normalize_keypoints(landmarks)

            # Hacer predicción si el número de keypoints es el esperado
            if len(keypoints) == 42:
                prediction = predict(model, keypoints)

                # Mostrar un cuadro de texto semitransparente en la parte inferior
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                # Mostrar la predicción en la ventana
                cv2.putText(frame, f'Detectado: {prediction}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar la ventana con la cámara
        cv2.imshow('Sign Language Detection', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
