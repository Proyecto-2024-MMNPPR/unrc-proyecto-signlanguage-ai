import os
import pickle
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic

# Constants
DATA_DIR = './data'
OUTPUT_PICKLE = 'data_normalized.pickle'
MAX_SEQUENCE_LENGTH = 30  # Fixed sequence length for all samples
NUM_KEYPOINTS = 21  # Number of keypoints per hand
NUM_COORDINATES = 2  # (x, y)
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDINATES * 2  # Features for both hands

def normalize_keypoints(landmarks):
    """Normalize hand keypoints relative to the minimum point."""
    if not landmarks:
        return [0] * (NUM_KEYPOINTS * NUM_COORDINATES)  # Return zeroed keypoints if no landmarks
    
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    normalized_landmarks = [
        (landmark.x - min_x) / (max_x - min_x) if max_x - min_x != 0 else 0,
        (landmark.y - min_y) / (max_y - min_y) if max_y - min_y != 0 else 0
        for landmark in landmarks
    ]
    return normalized_landmarks

def process_image(image_path, holistic):
    """Process a single image to extract and normalize hand keypoints."""
    image = cv2.imread(image_path)
    if image is None:
        return None  # Return None if the image cannot be loaded
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Get landmarks for both hands
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
    
    left_hand_keypoints = normalize_keypoints(left_hand_landmarks)
    right_hand_keypoints = normalize_keypoints(right_hand_landmarks)
    
    combined_keypoints = left_hand_keypoints + right_hand_keypoints
    return combined_keypoints if len(combined_keypoints) == NUM_FEATURES else None

def pad_or_trim_sequence(sequence, max_length, num_features):
    """Ensure all sequences have a fixed length by padding or trimming."""
    if len(sequence) < max_length:
        padding = [np.zeros(num_features) for _ in range(max_length - len(sequence))]
        sequence.extend(padding)
    return sequence[:max_length]  # Trim if sequence is longer than max length

def create_dataset():
    """Create a dataset from static and dynamic gesture images, save as pickle."""
    data, labels = [], []

    with Holistic(static_image_mode=True, min_detection_confidence=0.3) as holistic:
        for label in os.listdir(DATA_DIR):
            label_path = os.path.join(DATA_DIR, label)
            if not os.path.isdir(label_path):
                continue

            # Check if the label contains subdirectories (dynamic sequences)
            subdirectories = [d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d))]
            if subdirectories:
                # Process dynamic sequences
                for subdir in subdirectories:
                    sequence_path = os.path.join(label_path, subdir)
                    sequence_data = []

                    for img_name in sorted(os.listdir(sequence_path)):
                        img_path = os.path.join(sequence_path, img_name)
                        keypoints = process_image(img_path, holistic)
                        if keypoints:
                            sequence_data.append(keypoints)

                    sequence_data = pad_or_trim_sequence(sequence_data, MAX_SEQUENCE_LENGTH, NUM_FEATURES)
                    if len(sequence_data) == MAX_SEQUENCE_LENGTH:
                        data.append(sequence_data)
                        labels.append(label)
            else:
                # Process static images
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    keypoints = process_image(img_path, holistic)
                    if keypoints:
                        static_sequence = [keypoints] * MAX_SEQUENCE_LENGTH  # Replicate keypoints for static gestures
                        data.append(static_sequence)
                        labels.append(label)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    # Save normalized dataset using pickle
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Normalized dataset created and saved to '{OUTPUT_PICKLE}'.")

if __name__ == "__main__":
    create_dataset()
