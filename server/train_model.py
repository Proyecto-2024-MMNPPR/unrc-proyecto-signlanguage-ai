import pickle
import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Flatten
from tensorflow.keras.utils import to_categorical

# Load the normalized data
data_dict = pickle.load(open('./data_normalized.pickle', 'rb'))
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Convert labels to integers
unique_labels = np.unique(labels)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_to_label = {idx: label for label, idx in label_to_int.items()}
labels = np.array([label_to_int[label] for label in labels])

# Determine the maximum sequence length
max_sequence_length = 30  # Maximum number of frames per sequence
num_features = 84  # Number of features per frame (21 keypoints * 2 coordinates * 2 for x and y)

# Padding sequences to the maximum length
data_padded = np.array(data, dtype=np.float32)  # Convert to float32 for PyTorch

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert labels to one-hot encoding for the LSTM model
y_train_lstm = to_categorical(y_train, num_classes=len(unique_labels))
y_test_lstm = to_categorical(y_test, num_classes=len(unique_labels))

# -------- Random Forest --------
def train_random_forest(x_train, y_train, x_test, y_test):
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    model_rf = RandomForestClassifier()
    model_rf.fit(x_train_flat, y_train)
    y_predict_rf = model_rf.predict(x_test_flat)
    score_rf = accuracy_score(y_test, y_predict_rf)
    print('Random Forest Accuracy: {:.2f}%'.format(score_rf * 100))
    return model_rf, score_rf

# -------- LSTM --------
def train_lstm(x_train, y_train, x_test, y_test):
    model_lstm = Sequential()
    model_lstm.add(Input(shape=(max_sequence_length, num_features)))
    model_lstm.add(LSTM(64, return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(32))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(len(unique_labels), activation='softmax'))

    model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_lstm.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test), verbose=1)

    y_predict_lstm = np.argmax(model_lstm.predict(x_test), axis=1)
    score_lstm = accuracy_score(y_test.argmax(axis=1), y_predict_lstm)
    print('LSTM Accuracy: {:.2f}%'.format(score_lstm * 100))
    return model_lstm, score_lstm

def train_att_lstm(x_train, y_train, x_test, y_test):
    inputs = Input(shape=(max_sequence_length, num_features))
    lstm_out1 = LSTM(64, return_sequences=True)(inputs)
    lstm_out1 = Dropout(0.2)(lstm_out1)
    lstm_out2 = LSTM(64, return_sequences=True)(lstm_out1)
    lstm_out2 = Dropout(0.2)(lstm_out2)
    attention = Attention()([lstm_out2, lstm_out2])
    attention = Flatten()(attention)
    outputs = Dense(len(unique_labels), activation='softmax')(attention)

    model_att_lstm = Sequential()
    model_att_lstm = keras.Model(inputs=inputs, outputs=outputs)

    model_att_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_att_lstm.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test), verbose=1)

    
    y_predict_lstm = np.argmax(model_att_lstm.predict(x_test), axis=1)
    score_att_lstm = accuracy_score(y_test.argmax(axis=1), y_predict_lstm)
    print('LSTM + Attention Accuracy: {:.2f}%'.format(score_att_lstm * 100))
    return model_att_lstm, score_att_lstm

# Entrenar y evaluar ambos modelos
model_rf, score_rf = train_random_forest(x_train, y_train, x_test, y_test)
model_lstm, score_lstm = train_lstm(x_train, y_train_lstm, x_test, y_test_lstm)
model_att_lstm, score_att_lstm = train_att_lstm(x_train, y_train_lstm, x_test, y_test_lstm)

# Comparar las precisiones y guardar el mejor modelo
if score_rf >= score_lstm and score_rf >= score_att_lstm:
    best_model = model_rf
    model_name = 'Random Forest'
elif score_lstm >= score_rf and score_lstm >= score_att_lstm:
    best_model = model_lstm
    model_name = 'LSTM'
else:
    best_model = model_att_lstm
    model_name = 'LSTM + Attention'


# Save the best model
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model, 'model_type': model_name, 'label_map': int_to_label}, f)

print('The best model is {} with an accuracy of {:.2f}%'.format(model_name, max(score_rf, score_lstm, score_att_lstm) * 100))
