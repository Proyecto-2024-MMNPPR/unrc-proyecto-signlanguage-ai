import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tkinter import Tk, simpledialog
from helpers import create_folder, draw_keypoints, mediapipe_detection, there_hand
from constants import FONT, FONT_POS, FONT_SIZE

# Constants
DATA_DIR = './data'
RECORDING_DELAY_MS = 2000
CAPTURE_KEY_START = 's'
CAPTURE_KEY_FINISH = 'f'
CAPTURE_KEY_EXIT = 'q'
FRAME_SIZE = (480, 640, 3)
FONT_COLOR_RECORDING = (0, 255, 0)
FONT_COLOR_CAPTURING = (255, 50, 0)
FONT_COLOR_INSTRUCTIONS = (0, 255, 0)
DYNAMIC_REPETITIONS = 5
STATIC_REPETITIONS = 1

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_input_word():
    """
    Opens a dialog to get the word to capture and whether it's dynamic.
    Returns the word and whether it's dynamic.
    """
    root = Tk()
    root.withdraw()  # Hide main window

    # Ask for the word to capture
    word_to_train = simpledialog.askstring("Input", "Enter the word you want to capture:")
    if not word_to_train:
        return None, None

    # Replace spaces with underscores
    word_to_train = word_to_train.strip().replace(' ', '_').lower()

    # Validate input for dynamic or static word
    while True:
        is_dynamic = simpledialog.askstring("Input", f"Is '{word_to_train}' a dynamic word? (y/n):").strip().lower()
        if is_dynamic in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

    return word_to_train, is_dynamic == 'y'

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

            # Display a message before starting capture
            cv2.putText(image := np.zeros(FRAME_SIZE, dtype=np.uint8), f'Recording: {word}', (100, 240), FONT, 1.5, FONT_COLOR_RECORDING, 2)
            cv2.imshow(f'Capturing: {word}', image)
            cv2.waitKey(RECORDING_DELAY_MS)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image = frame.copy()
                results = mediapipe_detection(frame, holistic_model)

                # Display capture status on the screen
                status_text = f'Capturing: {len(frames)}' if capturing else 'Press "s" to start, "f" to finish'
                cv2.putText(image, status_text, FONT_POS, FONT, FONT_SIZE, FONT_COLOR_CAPTURING if capturing else FONT_COLOR_INSTRUCTIONS)
                draw_keypoints(image, results)
                cv2.imshow(f'Capturing: {word}', image)

                key = cv2.waitKey(10) & 0xFF

                # Start capturing
                if key == ord(CAPTURE_KEY_START) and not capturing:
                    capturing = True
                    frames = []  # Reset frames
                    print('Capture started...')

                # Stop capturing
                elif key == ord(CAPTURE_KEY_FINISH) and capturing:
                    capturing = False
                    print(f'Capture {rep + 1} finished.')
                    save_samples(class_dir, frames, rep, is_dynamic, word)
                    break

                # Collect frames only if a hand is detected
                if capturing and there_hand(results):
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

        # Prompt asking if the user wants to continue or stop
        response = simpledialog.askstring("Input", "Do you want to continue capturing another word? (y/n)").strip().lower()
        while response not in ['y', 'n']:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")
            response = simpledialog.askstring("Input", "Do you want to continue capturing another word? (y/n)").strip().lower()

        if response == 'y':
            main()  # Call main to capture a new word
        else:
            print("Capture process finished.")

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
    # Open a dialog to ask for the word and its type
    word_to_train, is_dynamic = get_input_word()
    if word_to_train:
        capture_samples(word_to_train, is_dynamic)

if __name__ == "__main__":
    main()
