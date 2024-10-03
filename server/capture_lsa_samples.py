import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def capture_samples(number_of_classes=5, dataset_size=100, margin_frame=1, min_cant_frames=5, delay_frames=3):
    '''
    ### CAPTURA DE MUESTRAS PARA LENGUAJE DE SEÑAS
    Captura imágenes para cada clase y las guarda en carpetas separadas.

    `number_of_classes` número de clases a capturar \n
    `dataset_size` cantidad de imágenes por clase \n
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
    `min_cant_frames` cantidad de frames mínimos para cada muestra \n
    `delay_frames` cantidad de frames que espera antes de detener la captura después de no detectar manos
    '''
    
    with Holistic() as holistic_model:
        cap = cv2.VideoCapture(0)

        for class_id in range(number_of_classes):
            class_dir = os.path.join(DATA_DIR, str(class_id))
            create_folder(class_dir)

            print(f'Colectando datos para la clase {class_id}')
            done = False

            # Esperar la señal para iniciar la captura
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, 'Ready? Press "Q" to start :)', (100, 50), FONT, FONT_SIZE, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) == ord('q'):
                    break

            count_frame = 0
            frames = []
            fix_frames = 0
            recording = False
            counter = 0

            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    break

                image = frame.copy()
                results = mediapipe_detection(frame, holistic_model)

                if there_hand(results) or recording:
                    recording = False
                    count_frame += 1
                    if count_frame > margin_frame:
                        cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                        frames.append(np.asarray(frame))
                else:
                    if len(frames) >= min_cant_frames + margin_frame:
                        fix_frames += 1
                        if fix_frames < delay_frames:
                            recording = True
                            continue
                        frames = frames[: - (margin_frame + delay_frames)]

                        # Guardar cada frame individualmente
                        for idx, frame in enumerate(frames):
                            cv2.imwrite(os.path.join(class_dir, f'{counter}_{idx}.jpg'), frame)

                        counter += 1
                        frames, count_frame = [], 0
                        continue

                    recording, fix_frames = False, 0
                    frames, count_frame = [], 0
                    cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0, 220, 100))

                draw_keypoints(image, results)
                cv2.imshow(f'Capturando clase {class_id}', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_samples(number_of_classes=5, dataset_size=100)
