import pickle
import numpy as np
import cv2
from constants import *
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS
from helpers import start_camera, start_window, there_hand

WINDOW_NAME = 'Sign Language Detection'

# Function to normalize keypoints
def normalize_keypoints(landmarks) -> list:
    """
    Normalize the x and y coordinates of a list of landmarks.

    This function takes a list of landmarks, extracts their x and y coordinates,
    and normalizes these coordinates to a range of [0, 1] based on the minimum
    and maximum values of the coordinates.

    Args:
        landmarks (list): A list of landmark objects, each having 'x' and 'y' attributes.

    Returns:
        list: A list of normalized x and y coordinates in the format
              [normalized_x1, normalized_y1, normalized_x2, normalized_y2, ...].
    """

    # Extract x and y coordinates
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]

    # Find min and max for normalization
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Normalize keypoints
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_x = (landmark.x - min_x) / (max_x - min_x) if max_x - min_x != 0 else 0
        normalized_y = (landmark.y - min_y) / (max_y - min_y) if max_y - min_y != 0 else 0
        normalized_landmarks.extend([normalized_x, normalized_y])

    return normalized_landmarks

# Function to extract keypoints from both hands
def extract_keypoints(results) -> list:
    """
    Extracts and normalizes keypoints from hand landmarks detected in the results.

    Args:
        results: An object containing hand landmark detection results. It should have
                 attributes `left_hand_landmarks` and `right_hand_landmarks`, each of which
                 can be either None or an object with a `landmark` attribute that is a list
                 of keypoints.

    Returns:
        list: A list of normalized keypoints. If a hand is not detected,
              the corresponding keypoints are set to zero.
    """

    # Retrieve landmarks for both hands
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else [np.zeros(21 * 2)]
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else [np.zeros(21 * 2)]

    # Normalize keypoints if hands are detected
    left_hand_keypoints = normalize_keypoints(left_hand_landmarks) if results.left_hand_landmarks else [0] * 42
    right_hand_keypoints = normalize_keypoints(right_hand_landmarks) if results.right_hand_landmarks else [0] * 42

    # Combine keypoints from both hands
    return left_hand_keypoints + right_hand_keypoints

# Function to make predictions based on a sequence of keypoints
def predict_sequence(model_data, sequence, max_sequence_length, num_features) -> str:
    """
    Predicts the label for a given sequence using the provided model.

    Parameters:
    model_data (dict): A dictionary containing the model, model type, and label map.
        - 'model': The trained model used for prediction.
        - 'model_type': The type of the model ('LSTM' or 'Random Forest').
        - 'label_map': A dictionary mapping label indices to label names.
    sequence (list or np.ndarray): The input sequence to be predicted.
    max_sequence_length (int): The maximum length of the sequence for LSTM models.
    num_features (int): The number of features in the sequence for LSTM models.

    Returns:
    str: The predicted label for the input sequence.
    """

    model = model_data['model']
    model_type = model_data['model_type']
    label_map = model_data['label_map']

    # Convert the sequence into a NumPy array
    sequence = np.array(sequence)

    # Adjust input shape based on model type
    if model_type == 'LSTM':
        sequence = sequence.reshape(1, max_sequence_length, num_features)
        prediction = model.predict(sequence)
        prediction = np.argmax(prediction, axis=1)
    elif model_type == 'Random Forest':
        sequence = sequence.flatten().reshape(1, -1)  # Flatten the sequence
        prediction = model.predict(sequence)

    return label_map[int(prediction[0])]


def perform_prediction(frame, model_data, results, sequence, max_sequence_length, num_features) -> str:
    """
    Perform prediction on a given video frame using a pre-trained model.

    Args:
        frame (numpy.ndarray): The current video frame.
        model_data (object): The pre-trained model data used for prediction.
        results (object): The results from a hand detection model containing landmarks.
        sequence (list): A list to store sequences of keypoints.
        max_sequence_length (int): The maximum length of the sequence for prediction.
        num_features (int): The number of features expected in the keypoints.

    Returns:
        object: The prediction result if the sequence is complete, otherwise None.
    """

    # Draw keypoints for both hands if detected
    if results.left_hand_landmarks:
        draw_landmarks(frame, results.left_hand_landmarks, HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        draw_landmarks(frame, results.right_hand_landmarks, HAND_CONNECTIONS)

    # Extract keypoints for both hands
    keypoints = extract_keypoints(results)

    # Append keypoints to the sequence if the correct length
    if len(keypoints) == num_features:
        sequence.append(keypoints)

        # Create a semi-transparent overlay for the bottom text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Control the sequence size, keeping only the last 'max_sequence_length' frames
    if len(sequence) > max_sequence_length:
        sequence.pop(0)

    # Make a prediction if the sequence is complete
    if len(sequence) == max_sequence_length:
        return predict_sequence(model_data, sequence, max_sequence_length, num_features)

def update_window(name, frame, msg) -> None:
    """
    Updates the given frame with a semi-transparent overlay and displays the prediction text.

    NOTE: This function will flip the frame horizontally before displaying it.

    Args:
        frame (numpy.ndarray): The current video frame to be updated.
        prediction (str): The predicted sign language text to be displayed on the frame.

    Returns:
        None
    """

    # Create a semi-transparent overlay for the bottom text
    frame = cv2.flip(frame, 1)  # Mirror the frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Display the prediction
    cv2.putText(frame, msg, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the camera window
    cv2.imshow(name, frame)
    cv2.pollKey()

def detect() -> None:
    """
    Main function to perform sign language detection using a trained model and MediaPipe Holistic.

    Dependencies:
        - pickle
        - cv2 (OpenCV)
        - mediapipe (MediaPipe Holistic)

    Returns:
        None
    """

    # Load the trained model and label mapping
    model_data = pickle.load(open(TRAINED_MODEL_FILE_NAME, 'rb'))
    if model_data:
        print("Model loaded successfully.")
    else:
        print("Error loading model.")
        print("Please ensure that the model file is located in the correct path.")
        print("Exiting...")
        exit(1)

    # Initialize MediaPipe Holistic
    with Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cam = start_camera()
        start_window(WINDOW_NAME)

        # List to store keypoint sequences
        sequence = []

        # Sequence parameters
        max_sequence_length = 30 # Fixed sequence length
        num_features = 84        # 21 points * 2 coordinates * 2 hands

        while cam.isOpened() and not cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            ret, frame = cam.read()
            if not ret:
                print("Error: Camera not accessible.")
                break


            # Process the image in RGB format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Perform prediction if hands are detected
            prediction = "(nothing)"
            if there_hand(results):
                prediction = perform_prediction(frame, model_data, results, sequence, max_sequence_length, num_features)

            # Update the prediction on the screen
            update_window(WINDOW_NAME, frame, f'Detected: {prediction}')

        print("Exiting...")
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()
