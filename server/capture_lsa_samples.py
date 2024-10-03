import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, ROOT_PATH

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Lista de palabras a capturar
WORDS_TO_CAPTURE = ['hola', 'bien', 'gracias', 'si', 'no']

def capture_samples(dataset_size=100, min_cant_frames=5):
    '''
    ### CAPTURA DE MUESTRAS PARA LENGUAJE DE SEÑAS
    Captura imágenes para cada palabra en WORDS_TO_CAPTURE y las guarda en carpetas separadas.

    `dataset_size` cantidad de imágenes por palabra \n
    `min_cant_frames` cantidad de frames mínimos para cada muestra \n
    '''
    
    with Holistic() as holistic_model:
        cap = cv2.VideoCapture(0)

        for word in WORDS_TO_CAPTURE:
            class_dir = os.path.join(DATA_DIR, word)
            create_folder(class_dir)

            print(f'Colectando datos para la palabra: {word}')
            capturing = False
            frames = []
            frame_scores = []  # Para almacenar la cantidad de puntos detectados por cada frame
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image = frame.copy()
                results = mediapipe_detection(frame, holistic_model)

                # Mostrar el estado en la pantalla
                if capturing:
                    cv2.putText(image, 'Capturando... Press "f" to finish', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                else:
                    cv2.putText(image, 'Press "s" to start capturing', FONT_POS, FONT, FONT_SIZE, (0, 255, 0))

                draw_keypoints(image, results)
                cv2.imshow(f'Capturando palabra: {word}', image)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):
                    # Iniciar captura
                    capturing = True
                    frames = []  # Limpiar frames previos
                    frame_scores = []  # Limpiar puntuaciones previas
                    print('Iniciando captura...')
                elif key == ord('f') and capturing:
                    # Finalizar captura
                    capturing = False
                    print('Finalizando captura...')
                    break
                elif key == ord('q'):
                    # Salir
                    capturing = False
                    print('Saliendo...')
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                # Captura los frames solo si se detecta una mano
                if capturing and there_hand(results):
                    frames.append(np.asarray(frame))
                    
                    # Puntuación basada en la cantidad de puntos detectados
                    num_points = 0
                    if results.left_hand_landmarks:
                        num_points += len(results.left_hand_landmarks.landmark)
                    if results.right_hand_landmarks:
                        num_points += len(results.right_hand_landmarks.landmark)

                    frame_scores.append((frame, num_points))

            # Seleccionar las 50 imágenes con mayor número de puntos detectados
            if len(frames) >= min_cant_frames:
                # Ordenar las imágenes por la cantidad de puntos detectados en orden descendente
                frame_scores.sort(key=lambda x: x[1], reverse=True)
                selected_frames = [frame for frame, _ in frame_scores[:50]]
                print(f'Guardando {len(selected_frames)} mejores imágenes en la carpeta {class_dir}')
                for idx, frame in enumerate(selected_frames):
                    cv2.imwrite(os.path.join(class_dir, f'{word}_{idx}.jpg'), frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_samples(dataset_size=100)
