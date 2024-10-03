import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Cargar los datos
data_dict = pickle.load(open('./data_normalized.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convertir etiquetas a números enteros
unique_labels = np.unique(labels)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_to_label = {idx: label for label, idx in label_to_int.items()}
labels = np.array([label_to_int[label] for label in labels])

# Dividir el conjunto de datos
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convertir etiquetas para el modelo LSTM
y_train_lstm = to_categorical(y_train, num_classes=len(unique_labels))
y_test_lstm = to_categorical(y_test, num_classes=len(unique_labels))

# -------- Random Forest --------
def train_random_forest(x_train, y_train, x_test, y_test):
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train, y_train)
    y_predict_rf = model_rf.predict(x_test)
    score_rf = accuracy_score(y_test, y_predict_rf)
    print('Random Forest Accuracy: {:.2f}%'.format(score_rf * 100))
    return model_rf, score_rf

# -------- LSTM --------
def train_lstm(x_train, y_train, x_test, y_test):
    # Redefinir las dimensiones para LSTM (samples, timesteps, features)
    x_train_lstm = np.expand_dims(x_train, axis=1)
    x_test_lstm = np.expand_dims(x_test, axis=1)

    model_lstm = Sequential()
    model_lstm.add(LSTM(64, input_shape=(1, x_train.shape[1]), return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(32))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(len(unique_labels), activation='softmax'))

    model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_lstm.fit(x_train_lstm, y_train, epochs=20, batch_size=16, validation_data=(x_test_lstm, y_test), verbose=0)

    y_predict_lstm = np.argmax(model_lstm.predict(x_test_lstm), axis=1)
    score_lstm = accuracy_score(y_test.argmax(axis=1), y_predict_lstm)
    print('LSTM Accuracy: {:.2f}%'.format(score_lstm * 100))
    return model_lstm, score_lstm

# Entrenar y evaluar ambos modelos
model_rf, score_rf = train_random_forest(x_train, y_train, x_test, y_test)
model_lstm, score_lstm = train_lstm(x_train, y_train_lstm, x_test, y_test_lstm)

# Comparar las precisiones y guardar el mejor modelo
if score_rf >= score_lstm:
    best_model = model_rf
    model_name = 'Random Forest'
else:
    best_model = model_lstm
    model_name = 'LSTM'

# Guardar el modelo seleccionado
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model, 'model_type': model_name, 'label_map': int_to_label}, f)

print('El mejor modelo es {} con una precisión de {:.2f}%'.format(model_name, max(score_rf, score_lstm) * 100))
