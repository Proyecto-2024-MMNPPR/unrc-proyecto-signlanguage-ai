import os
import cv2
import numpy as np
from PyQt5 import QtWidgets
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection
from constants import FONT, FONT_POS, FONT_SIZE

# Constants
DATA_DIR = './data'
RECORDING_DELAY_MS = 2000
CAPTURE_KEY_START = 's'
CAPTURE_KEY_FINISH = 'f'
CAPTURE_KEY_EXIT = 'q'
CAPTURE_KEY_PAUSE = 'p'
FRAME_SIZE = (480, 640, 3)
FONT_COLOR_RECORDING = (0, 255, 0)
FONT_COLOR_CAPTURING = (255, 50, 0)
FONT_COLOR_PAUSED = (255, 255, 0)
FONT_COLOR_INSTRUCTIONS = (0, 255, 0)
DYNAMIC_REPETITIONS = 5
STATIC_REPETITIONS = 1

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_input_word():
    """
    Prompts the user to input the word and whether it is dynamic via the console.
    Returns the word and whether it's dynamic.
    """
    word_to_train = input("Enter the word you want to capture: ").strip().replace(' ', '_').lower()

    # Ensure valid input for dynamic word question
    while True:
        is_dynamic = input("Is this a dynamic word? (y/n): ").lower()
        if is_dynamic in ['y', 'n']:
            is_dynamic = is_dynamic == 'y'
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    
    return word_to_train, is_dynamic

def capture_samples(word, is_dynamic=False, dataset_size=100, sequence_length=30):
    """
    Captures samples for the given word using webcam video.
    `word`: The word to be captured.
    `is_dynamic`: Determines if the word is dynamic (requires multiple repetitions).
    """
    with Holistic() as holistic_model:
        cap = cv2.VideoCapture(0)

        # Create directory for the word if it doesn't exist
        class_dir = os.path.join(DATA_DIR, word)
        create_folder(class_dir)

        repetitions = DYNAMIC_REPETITIONS if is_dynamic else STATIC_REPETITIONS
        num_frames = sequence_length if is_dynamic else dataset_size

        for rep in range(repetitions):
            print(f"Starting capture {rep + 1} of {repetitions} for '{word}'")
            frames = []
            capturing = False
            paused = False

            # Display a message before starting capture
            cv2.putText(image := np.zeros(FRAME_SIZE, dtype=np.uint8), f'Recording: {word}', (100, 240), FONT, 1.5, FONT_COLOR_RECORDING, 2)
            cv2.imshow(f'Capturing: {word}', image)
            cv2.waitKey(RECORDING_DELAY_MS)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Camera not accessible.")
                    break

                image = frame.copy()
                results = mediapipe_detection(frame, holistic_model)

                # Display capture status on the screen
                if paused:
                    status_text = 'Paused. Press "p" to resume'
                    font_color = FONT_COLOR_PAUSED
                elif capturing:
                    status_text = f'Capturing: {len(frames)}'
                    font_color = FONT_COLOR_CAPTURING
                else:
                    status_text = 'Press "s" to start, "f" to finish, "p" to pause'
                    font_color = FONT_COLOR_INSTRUCTIONS

                cv2.putText(image, status_text, FONT_POS, FONT, FONT_SIZE, font_color)
                draw_keypoints(image, results)
                cv2.imshow(f'Capturing: {word}', image)

                key = cv2.waitKey(10) & 0xFF

                # Start capturing
                if key == ord(CAPTURE_KEY_START) and not capturing and not paused:
                    capturing = True
                    frames = []  # Reset frames
                    print('Capture started...')

                # Pause capturing
                elif key == ord(CAPTURE_KEY_PAUSE):
                    paused = not paused
                    if paused:
                        print('Capture paused...')
                    else:
                        print('Capture resumed...')

                # Stop capturing
                elif key == ord(CAPTURE_KEY_FINISH) and capturing:
                    capturing = False
                    print(f'Capture {rep + 1} finished.')
                    save_samples(class_dir, frames, rep, is_dynamic, word)
                    break

                # Collect frames regardless of whether a hand is detected
                if capturing and not paused:
                    frames.append(frame)

                # Stop capturing when the required number of frames is collected
                if not is_dynamic and len(frames) >= num_frames:
                    capturing = False
                    print(f'Capture {rep + 1} finished.')
                    save_samples(class_dir, frames, rep, is_dynamic, word)
                    break

                # Exit on 'q' key
                if key == ord(CAPTURE_KEY_EXIT):
                    print('Exiting...')
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Delay before starting next repetition for dynamic words
            if is_dynamic and rep < repetitions - 1:
                print(f"Capture {rep + 1} complete. Preparing for the next...")
                cv2.waitKey(RECORDING_DELAY_MS)

        cap.release()
        cv2.destroyAllWindows()

        # Ask the user if they want to capture another word
        while True:
            response = input("Do you want to capture another word? (y/n): ").lower()
            if response in ['y', 'n']:
                if response == 'y':
                    main()
                else:
                    print("Capture process finished.")
                break
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")

def save_samples(class_dir, frames, rep, is_dynamic, word):
    """
    Saves captured frames as images. For dynamic words, saves them in a sequence.
    """
    if is_dynamic:
        sequence_folder = os.path.join(class_dir, f'sequence_{rep + 1}')
        create_folder(sequence_folder)
        for idx, frame in enumerate(frames):
            frame_path = os.path.join(sequence_folder, f'frame_{idx}.jpg')
            cv2.imwrite(frame_path, frame)
            # Save mirrored frame
            mirrored_frame = cv2.flip(frame, 1)
            mirrored_path = os.path.join(sequence_folder, f'frame_{idx}_mirror.jpg')
            cv2.imwrite(mirrored_path, mirrored_frame)
    else:
        for idx, frame in enumerate(frames):
            original_path = os.path.join(class_dir, f'{word}_{idx}.jpg')
            cv2.imwrite(original_path, frame)
            # Save mirrored frame
            mirrored_frame = cv2.flip(frame, 1)
            mirrored_path = os.path.join(class_dir, f'{word}_{idx}_mirror.jpg')
            cv2.imwrite(mirrored_path, mirrored_frame)

def main():
    # Get input from the user via the console
    word_to_train, is_dynamic = get_input_word()
    if word_to_train:
        capture_samples(word_to_train, is_dynamic)

if __name__ == "__main__":
    main()
