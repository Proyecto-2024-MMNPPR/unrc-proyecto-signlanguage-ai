import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ai', 'model', 'model.p')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'ai', 'model', 'label_dict.pickle')
with open(MODEL_PATH, 'rb') as model_file:
    model_data = pickle.load(model_file)

with open(LABELS_PATH, 'rb') as labels_file:
    label_map = pickle.load(labels_file)

app = Flask(__name__)
CORS(app)

MAX_SEQUENCE_LENGTH = 5
NUM_FEATURES = 96

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = data.get('sequence')

    if sequence is None or len(sequence) == 0:
        return jsonify({'error': 'No sequence provided'}), 400

    for i, frame in enumerate(sequence):
        if len(frame) != NUM_FEATURES:
            return jsonify({'error': f'Frame {i} has an incorrect number of features. Expected {NUM_FEATURES}, but got {len(frame)}'}), 400
        print(f"Frame {i}: Length = {len(frame)}, Type = {type(frame)}")

    try:
        sequence_np = np.array(sequence)
        print(f"Shape of sequence: {sequence_np.shape}")
    except ValueError as e:
        print(f"Error in sequence format: {str(e)}")
        return jsonify({'error': f'Error in sequence format: {str(e)}'}), 400

    if sequence_np.shape[0] != MAX_SEQUENCE_LENGTH or sequence_np.shape[1] != NUM_FEATURES:
        return jsonify({'error': 'Invalid sequence length or features'}), 400

    try:
        prediction = predict_sequence(model_data, label_map, sequence_np, MAX_SEQUENCE_LENGTH, NUM_FEATURES)
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_sequence(model_data, labels_data, sequence, max_sequence_length, num_features):
    model = model_data
    model_type = 'Random Forest'
    label_map = labels_data

    if model_type == 'LSTM' or model_type == 'LSTM + Attention':
        sequence = sequence.reshape(1, max_sequence_length, num_features)
        prediction = model.predict(sequence)
        prediction = np.argmax(prediction, axis=1)
    elif model_type == 'Random Forest':
        sequence = sequence.flatten().reshape(1, -1)
        prediction = model.predict(sequence)

    return label_map[int(prediction[0])]

if __name__ == '__main__':
    app.run(debug=True)
