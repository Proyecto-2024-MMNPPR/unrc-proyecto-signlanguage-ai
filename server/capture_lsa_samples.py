import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, there_hand
from constants import FONT, FONT_POS, FONT_SIZE

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Definimos las palabras estáticas y dinámicas
static_words = ['a', 'e', 'i', 'o', 'u', 'por favor', 'hola', 'yo','tomar']  # Añadimos "por favor" a la lista de palabras estáticas
dynamic_words = ['quedate', 'casa']

def capture_samples(words_to_capture=None, dataset_size=50, sequence_length=30):
    '''
    ### CAPTURA DE MUESTRAS PARA LENGUAJE DE SEÑAS
    Captura palabras estáticas y dinámicas automáticamente.
    `words_to_capture`: Lista de palabras que se desea capturar. Si es None, captura todas.
    '''
    with Holistic() as holistic_model:
        cap = cv2.VideoCapture(0)

        # Si no se especifican palabras, se capturan todas
        if words_to_capture is None:
            words_to_capture = static_words + dynamic_words

        for word in words_to_capture:
            # Crear carpeta para guardar la palabra/letra
            class_dir = os.path.join(DATA_DIR, word)
            
            # Verificar si la palabra ya tiene datos capturados
            if os.path.exists(class_dir):
                print(f"'{word}' ya tiene capturas. ¿Deseas capturarla de nuevo? (s/n): ")
                choice = input().strip().lower()
                if choice != 's':
                    print(f"Saltando '{word}'.")
                    continue
            else:
                create_folder(class_dir)

            # Determinar si la palabra es dinámica
            is_dynamic = word in dynamic_words
            repetitions = 5 if is_dynamic else 1
            num_frames = 30 if is_dynamic else dataset_size

            for rep in range(repetitions):
                print(f"Iniciando captura {rep + 1} de {repetitions} para '{word}'")
                frames = []
                capturing = False

                # Mostrar el nombre de la palabra en pantalla antes de comenzar
                cv2.putText(image := np.zeros((480, 640, 3), dtype=np.uint8), f'Grabando: {word}', (100, 240), FONT, 1.5, (0, 255, 0), 2)
                cv2.imshow(f'Capturando palabra/letra: {word}', image)
                cv2.waitKey(2000)  # Pausa de 2 segundos antes de empezar la captura

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image = frame.copy()
                    results = mediapipe_detection(frame, holistic_model)

                    # Mostrar el estado en la pantalla
                    if capturing:
                        cv2.putText(image, f'Capturando: {len(frames)}', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    else:
                        cv2.putText(image, 'Presiona "s" para iniciar, "f" para finalizar', FONT_POS, FONT, FONT_SIZE, (0, 255, 0))

                    draw_keypoints(image, results)
                    cv2.imshow(f'Capturando palabra/letra: {word}', image)

                    key = cv2.waitKey(10) & 0xFF

                    # Iniciar la captura
                    if key == ord('s') and not capturing:
                        capturing = True
                        frames = []  # Reiniciar frames
                        print('Iniciando captura...')

                    # Finalizar la captura
                    elif key == ord('f') and capturing:
                        capturing = False
                        print(f'Finalizando captura {rep + 1}...')

                        # Guardar secuencia dinámica
                        if is_dynamic:
                            sequence_folder = os.path.join(class_dir, f'sequence_{rep + 1}')
                            create_folder(sequence_folder)
                            for idx, frame in enumerate(frames):
                                frame_path = os.path.join(sequence_folder, f'frame_{idx}.jpg')
                                cv2.imwrite(frame_path, frame)

                                # Crear y guardar imagen espejo
                                mirrored_frame = cv2.flip(frame, 1)
                                mirrored_path = os.path.join(sequence_folder, f'frame_{idx}_mirror.jpg')
                                cv2.imwrite(mirrored_path, mirrored_frame)
                        break

                    # Guardar imágenes en caso de alcanzar el número necesario para palabras estáticas
                    if not is_dynamic and len(frames) >= num_frames:
                        capturing = False
                        print(f'Finalizando captura {rep + 1}...')

                        # Guardar imágenes estáticas
                        for idx, frame in enumerate(frames):
                            original_path = os.path.join(class_dir, f'{word}_{idx}.jpg')
                            cv2.imwrite(original_path, frame)

                            # Crear y guardar imagen espejo
                            mirrored_frame = cv2.flip(frame, 1)
                            mirrored_path = os.path.join(class_dir, f'{word}_{idx}_mirror.jpg')
                            cv2.imwrite(mirrored_path, mirrored_frame)
                        break

                    # Captura los frames solo si se detecta una mano
                    if capturing and there_hand(results):
                        frames.append(frame)

                    # Salir con 'q'
                    if key == ord('q'):
                        print('Saliendo...')
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Añadir un breve retraso entre repeticiones en modo dinámico
                if is_dynamic and rep < repetitions - 1:
                    print(f"Captura {rep + 1} completada. Preparando para la siguiente...")
                    cv2.waitKey(2000)  # Esperar 2 segundos antes de la siguiente captura

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Puedes especificar las palabras que deseas capturar en la lista:
    words_to_capture = ['tomar']  # Añade "por favor" u otras palabras aquí para capturarlas
    capture_samples(words_to_capture)
