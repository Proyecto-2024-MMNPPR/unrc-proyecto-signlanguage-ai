from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import time

app = Flask(__name__)
CORS(app, resources={
    r"/process-frame": {
        "origins": [
            "http://localhost:5173",
            "https://signlanguage-ai.vercel.app",
            "https://backend-signlanguage-ai.vercel.app"
        ]
    }
})

@app.route('/api', methods=['GET'])
def index():
    return 'Hello, World!'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

last_message = ''
last_message_time = 0

def label(idx, hand, results):
    aux = None
    for _, clase in enumerate(results.multi_handedness):
        if clase.classification[0].index == idx:
            label = clase.classification[0].label
            texto = '{}'.format(label)
            coords = tuple(np.multiply(np.array(
                (hand.landmark[mp_hands.HandLandmark.WRIST].x, 
                 hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [1920, 1080]).astype(int))
            aux = texto, coords
    return aux

def euclidian_distance(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

def process_hand_landmarks(image, results):
    global last_message, last_message_time
    image_height, image_width, _ = image.shape
    response_message = ''
    change = True
    change2 = False

    if results.multi_hand_landmarks:
        for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
            index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                int(hand_landmarks.landmark[6].y * image_height))
            
            thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                         int(hand_landmarks.landmark[4].y * image_height))
            thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                         int(hand_landmarks.landmark[2].y * image_height))
            
            middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                 int(hand_landmarks.landmark[12].y * image_height))
            
            pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                         int(hand_landmarks.landmark[20].y * image_height))
            
            wrist = (int(hand_landmarks.landmark[0].x * image_width),
                     int(hand_landmarks.landmark[0].y * image_height))
            
            if thumb_pip[1] - thumb_tip[1] > 0 and thumb_pip[1] - index_finger_tip[1] < 0 \
                and thumb_pip[1] - middle_finger_tip[1] < 0 and thumb_pip[1] - pinky_tip[1] < 0:
                response_message = 'Bien'
            elif thumb_pip[1] - thumb_tip[1] < 0 and thumb_pip[1] - index_finger_tip[1] > 0 \
                and thumb_pip[1] - middle_finger_tip[1] > 0 and thumb_pip[1] - pinky_tip[1] > 0:
                response_message = 'Mal'
            elif thumb_pip[1] - thumb_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] > 0 \
                and pinky_tip[1] < wrist[1]:
                response_message = 'Te amo'
            if label(num, hand_landmarks, results) and len(results.multi_hand_landmarks) == 2:
                text, coords = label(num, hand_landmarks, results)
                if text == "Right":
                    change = True
                if text == "Left":
                    change2 = True

                if change and change2:
                    if euclidian_distance(index_finger_tip, wrist) < 170.0:
                        response_message = '¿Qué hora es?'
    
    current_time = time.time()
    if response_message == last_message and (current_time - last_message_time < 5):
        response_message = ''

    if response_message:
        last_message = response_message
        last_message_time = current_time

    return response_message

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    data = request.json
    img_base64 = data.get('image', None)
    
    if img_base64 is None:
        return jsonify({'message': 'No se recibió ninguna imagen.'}), 400

    # Decodifica la imagen base64
    img_data = base64.b64decode(img_base64)
    np_img = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2) as hands:
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        response_message = process_hand_landmarks(image, results)

    _, buffer = cv2.imencode('.jpg', image)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'message': response_message, 'processed_image': processed_img_base64})

if __name__ == '__main__':
    app.run(debug=True)
