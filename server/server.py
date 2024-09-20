from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import cv2
import mediapipe as mp
import base64
from flask import Flask, request, jsonify
from website import *

app = create_app()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    img_data = base64.b64decode(data)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        predicted_index = np.argmax(prediction_proba)
        confidence = prediction_proba[0][predicted_index] * 100
        predicted_character = labels_dict[int(predicted_index)]

        response = {
            'message': predicted_character,
            'confidence': confidence,
            'processed_image': base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
        }
        return jsonify(response)

    return jsonify({'message': 'No hand detected', 'processed_image': '', 'confidence': 0})

@app.route('/login', methods = ['POST'])
def login():
    if request.method == 'POST':
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
