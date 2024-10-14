import os
import pickle
import numpy as np
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS
from tts import text_to_speech

# Load the trained model and label mapping
model_data = pickle.load(open('model.p', 'rb'))
model = model_data['model']
model_type = model_data['model_type']
label_map = model_data['label_map']

# Sequence parameters
max_sequence_length = 30  # Fixed sequence length
num_features = 84  # 21 points * 2 coordinates * 2 hands

# Function to normalize keypoints
def normalize_keypoints(landmarks):
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
def extract_keypoints(results):
    # Retrieve landmarks for both hands
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else [np.zeros(21 * 2)]
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else [np.zeros(21 * 2)]

    # Normalize keypoints if hands are detected
    left_hand_keypoints = normalize_keypoints(left_hand_landmarks) if results.left_hand_landmarks else [0] * 42
    right_hand_keypoints = normalize_keypoints(right_hand_landmarks) if results.right_hand_landmarks else [0] * 42

    # Combine keypoints from both hands
    return left_hand_keypoints + right_hand_keypoints

# Function to make predictions based on a sequence of keypoints
def predict_sequence(model, sequence):
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

# Initialize MediaPipe Holistic
with Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit")

    # List to store keypoint sequences
    sequence = []
    last_prediction = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the image in RGB format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

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
            prediction = predict_sequence(model, sequence)

            # Show the detected word if different from the last one
            if prediction != last_prediction:
                last_prediction = prediction

            # Create a semi-transparent overlay for the bottom text
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Display the prediction
            cv2.putText(frame, f'Detected: {last_prediction}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the camera window
        cv2.imshow('Sign Language Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
