import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'ai', 'model', 'model.p')

# Cargar el modelo entrenado y el mapa de etiquetas
model_data = pickle.load(open(model_path, 'rb'))
model = model_data['model']
model_type = model_data['model_type']
label_map = model_data['label_map']

# Parámetros de la secuencia
max_sequence_length = 30  # Longitud fija de la secuencia
num_features = 84  # 21 puntos * 2 coordenadas * 2 manos

# Función para hacer predicciones basadas en una secuencia de keypoints
def predict_sequence(model, sequence):
    sequence = np.array(sequence)

    if len(sequence[0]) != num_features:
        if len(sequence[0]) == 42:
            sequence = [np.concatenate([frame, np.zeros(42)]) for frame in sequence]  # Rellenar con 42 ceros
        else:
            raise ValueError("Each frame must have either 42 or 84 features")

    # Ajustar la forma de la entrada según el tipo de modelo
    if model_type == 'LSTM':
        sequence = sequence.reshape(1, max_sequence_length, num_features)
        prediction = model.predict(sequence)
        prediction = np.argmax(prediction, axis=1)
    elif model_type == 'Random Forest':
        sequence = sequence.flatten().reshape(1, -1)
        prediction = model.predict(sequence)
    
    return label_map[int(prediction[0])]

# Ruta para recibir las secuencias de keypoints y realizar una predicción
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = data.get('sequence', [])

    # Asegurarse de que la secuencia tenga la longitud correcta
    if len(sequence) != max_sequence_length:
        return jsonify({'error': 'Invalid sequence length'}), 400

    prediction = predict_sequence(model, sequence)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
