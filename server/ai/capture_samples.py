import os
import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
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


def capture_samples(word: str, sequence_length: int = 5) -> None:
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

        start_window(capturing_msg)

        for rep in range(repetitions):
            cam = start_camera()
            frames = []
            capturing = False
            paused = False
            prev_frame = None

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
                    prev_frame = None
                    print('Capture started...')

                # Pause capturing
                elif key == ord(PAUSE_KEY):
                    paused = not paused
                    if paused:
                        print('Capture paused...')
                    else:
                        print('Capture resumed...')

                # Stop capturing
                elif capturing and key == ord(FINISH_KEY):
                    capturing = False
                    print(f'Capture {rep + 1}/{repetitions} for “{word}” finished.')
                    save_samples(word_dir, frames, rep)
                    update_window(capturing_msg, image, f'Capture {rep + 1}/{repetitions} finished.')
                    time.sleep(0.75) # Give user a moment to see the message
                    break

                # Collect frames regardless of whether a hand is detected
                if capturing and not paused:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = gaussian_filter(gray, sigma=1.5)

                    if prev_frame is not None:
                        diff = cv2.absdiff(prev_frame, gray)
                        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                        keypoints = np.argwhere(thresh > 0)

                        if keypoints.size > 0:
                            frames.append(frame)

                    prev_frame = gray

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
    Saves 5 significant frames from the capture sequence.
    """

    example_folder = os.path.join(word_dir, f'example_{rep + 1}')
    create_folder(example_folder)
    num_frames = NUM_REPRESENTATIVE_FRAMES
    selected_frames = np.linspace(0, len(frames) - 1, num_frames, dtype=int)

    for idx, frame_idx in enumerate(selected_frames):
        frame_path = os.path.join(example_folder, f'sequence_{idx + 1}.jpg')
        cv2.imwrite(frame_path, frames[frame_idx])


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
