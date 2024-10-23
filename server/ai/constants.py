import os
from cv2 import FONT_HERSHEY_DUPLEX

AI_PWD = os.path.dirname(os.path.abspath(__file__))


# Settings
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15
TRAINED_MODEL_FILE_NAME = f"{AI_PWD}/model/model.p"

# Sample capture
RECORDING_DELAY_MS = 2000
START_KEY = 's'
FINISH_KEY = 'f'
EXIT_KEY = 'q'
PAUSE_KEY = 'p'
DYNAMIC_REPETITIONS = 20 # n_examples
NUM_REPRESENTATIVE_FRAMES = 5

# Data Set Creation
DATA_DIR = f'{AI_PWD}/model/data'
OUTPUT_PICKLE = f'{AI_PWD}/model/data_normalized.pickle'
MAX_SEQUENCE_LENGTH = 30  # Fixed sequence length for all samples
NUM_KEYPOINTS = 21  # Number of keypoints per hand
NUM_COORDINATES = 2  # (x, y)
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDINATES * 2  # Features for both hands


# Show image parameters
FONT = FONT_HERSHEY_DUPLEX
FONT_SIZE = 1.5
FONT_POS = (5, 30)
