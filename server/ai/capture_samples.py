import os
import time
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *


def get_input_word() -> str:
    """
    Prompts the user to input the word and whether it is dynamic via the console.
    Returns the word.
    """

    word_to_train = input("Enter the word you want to capture: ").strip().replace(' ', '_').lower()

    return word_to_train


def capture_samples(word: str, sequence_length: int = 30) -> None:
    """
    Captures samples for the given word using webcam video.
    `word`: The word to be captured.
    """

    with Holistic() as holistic_model:
        # Create directory for the word if it doesn't exist
        word_dir = os.path.join(DATA_DIR, word)
        create_folder(word_dir)

        capturing_msg = f'Capturing: "{word}"'
        repetitions = DYNAMIC_REPETITIONS
        num_frames = sequence_length

        start_window(capturing_msg)

        for rep in range(repetitions):
            cam = start_camera()

            frames = []
            capturing = False
            paused = False

            print(f"Starting capture {rep + 1}/{repetitions} for “{word}”")

            while True:
                ret, frame = cam.read()
                if not ret:
                    report_error("Camera not accessible.")
                    exit(1)

                image = frame.copy()
                results = mediapipe_detection(frame, holistic_model)
                draw_keypoints(image, results)

                # Display capture status on the screen
                if paused:
                    status_text = 'Paused. Press "p" to resume'
                elif capturing:
                    status_text = f'Capturing: {len(frames)} frames'
                else:
                    status_text = f'Capturing: {rep + 1}/{repetitions} for "{word}"' + '\nPress "s" to start, "f" to finish, "p" to pause, "q" to exit'

                key = update_window(capturing_msg, image, status_text)

                # Start capturing
                if key == ord(START_KEY) and not capturing and not paused:
                    capturing = True
                    frames = []  # Reset frames
                    print('Capture started...')

                # Pause capturing
                elif key == ord(PAUSE_KEY):
                    paused = not paused
                    if paused:
                        print('Capture paused...')
                    else:
                        print('Capture resumed...')

                # Stop capturing
                elif capturing and key == ord(FINISH_KEY) or len(frames) >= num_frames:
                    capturing = False
                    print(f'Capture {rep + 1}/{repetitions} for “{word}” finished.')
                    save_samples(word_dir, frames, rep)
                    update_window(capturing_msg, image, f'Capture {rep + 1}/{repetitions} finished.')
                    time.sleep(0.75) # Give user a moment to see the message
                    break

                # Collect frames regardless of whether a hand is detected
                if capturing and not paused:
                    frames.append(frame)

                # Exit on 'q' key or if the window is closed
                if cv2.getWindowProperty(capturing_msg, cv2.WND_PROP_VISIBLE) < 1 or key == ord(EXIT_KEY):
                    print('Exiting...')
                    cam.release()
                    cv2.destroyAllWindows()
                    exit(0)


            cam.release()

        cv2.destroyAllWindows()


def save_samples(word_dir: str, frames, rep: int) -> None:
    """
    Saves captured frames as images in a sequence.
    """

    sequence_folder = os.path.join(word_dir, f'sequence_{rep + 1}')
    create_folder(sequence_folder)
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(sequence_folder, f'frame_{idx}.jpg')
        cv2.imwrite(frame_path, frame)

        # Save mirrored frame
        mirrored_frame = cv2.flip(frame, 1)
        mirrored_path = os.path.join(sequence_folder, f'frame_{idx}_mirror.jpg')
        cv2.imwrite(mirrored_path, mirrored_frame)


def want_to_capture_another_word() -> bool:
    """
    Prompts the user if they want to capture another word.
    Returns True if the user wants to capture another word, False otherwise.
    """

    response = input("Do you want to capture another word? [(default) y, n]: ").lower()
    if response in ['y', 'n', '']:
        if response == 'y' or response == '':
            return True
        else:
            print("Capture process finished.")
            return False
    else:
        print("Invalid input. Please enter 'y' for yes or 'n' for no.")
        return want_to_capture_another_word()


def main() -> None:
    # Ensure data directory exists
    create_folder(DATA_DIR)

    finish = False

    while not finish:
        word_to_train = get_input_word()
        if word_to_train:
            capture_samples(word_to_train)

        finish = not want_to_capture_another_word()

if __name__ == "__main__":
    main()
