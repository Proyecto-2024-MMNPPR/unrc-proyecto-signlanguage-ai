import json
import os
import cv2
from cv2 import VideoCapture
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
import numpy as np
import pandas as pd
from typing import NamedTuple
from constants import *
import sys

def report_error(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks

def get_word_ids(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data.get('word_ids')

def start_camera() -> VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap

def start_window(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(name, 100, 100)
    print("Close the window to exit.")

def update_window(name: str, frame, msg: str, fg_color: tuple = (255, 255, 255), bg_color: tuple = (0, 0, 0)) -> int:
    """
    Updates the given frame with a semi-transparent overlay and displays the prediction text.

    Args:
        frame (numpy.ndarray): The current video frame to be updated.
        prediction (str): The predicted sign language text to be displayed on the frame.

    Returns:
        int: The key code of the pressed key, because cv2 windows need to be interacted with to update the frame.
    """

    # Split the message into multiple lines if necessary
    max_width = frame.shape[1] - 20
    words = msg.split(' ')
    lines = []
    current_line = words[0]

    for word in words[1:]:
        if '\n' in word:
            split_words = word.split('\n')
            for split_word in split_words[:-1]:
                if cv2.getTextSize(current_line + ' ' + split_word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] < max_width:
                    current_line += ' ' + split_word
                else:
                    lines.append(current_line)
                    current_line = split_word
            lines.append(current_line)
            current_line = split_words[-1]
        else:
            if cv2.getTextSize(current_line + ' ' + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] < max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
    lines.append(current_line)

    # Create a semi-transparent overlay for the bottom text
    frame = cv2.flip(frame, 1)  # Mirror the frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - 20 * (len(lines) + 1) - 30), (frame.shape[1], frame.shape[0]), bg_color, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Display the message
    y0, dy = frame.shape[0] - 20 * len(lines) - 15 , 35
    for i, line in enumerate(lines):
        # print(y0, i + 1, dy, y0 + i * dy)
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, fg_color, 2, cv2.LINE_AA)

    # Show the camera window
    cv2.imshow(name, frame)

    # Flush the buffer to show the frame
    return cv2.pollKey() & 0xFF


def draw_keypoints(image, results: NamedTuple) -> None:
    """
    Draws keypoints and connections on the given image based on the provided results.

    Parameters:
    image (cv2.MatLike): The image on which to draw the keypoints and connections.
    results (NamedTuple): A named tuple containing the landmarks for face, pose, left hand, and right hand.

    Returns:None
    """

    # Draw face connections
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    # Draw pose connections
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    # Draw left hand connections
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Draw right hand connections
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

def save_frames(frames, output_folder: str) -> None:
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
