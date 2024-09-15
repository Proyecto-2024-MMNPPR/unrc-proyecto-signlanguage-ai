# ✋ LSA Sign Language Detection Project

### 🇦🇷 Universidad Nacional de Río Cuarto (UNRC), Argentina

This project aims to develop a machine learning model to detect and classify the signs of the Argentine Sign Language (LSA) alphabet. The focus is on recognizing the letters of the alphabet, particularly the vowels, using computer vision techniques like **OpenCV** and **MediaPipe**, and training a classifier using **RandomForestClassifier**. The backend is built using **React**.

## 🌟 Project Overview

The goal of this project is to facilitate communication for people with hearing impairments by providing a tool that can translate LSA gestures into text. This tool uses real-time video input to capture hand gestures and translate them into corresponding letters of the LSA alphabet.

### 💻 Technologies Used

- **Python** 🐍 for the core programming language.
- **OpenCV** 📸 for image capture and processing.
- **MediaPipe** 🖐️ for hand landmark detection.
- **scikit-learn** 📊 for training the machine learning model.
- **RandomForestClassifier** 🌲 as the primary model for classification.
- **React JS** ⚛️ for building the user interface and backend.

### 👥 Team Members

This project is developed by students and professors from the Universidad Nacional de Río Cuarto (UNRC) as part of the **Computer Science Program**.

## 🚀 How to Use

### Requirements

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

### Data Collection

You can collect images for each letter of the alphabet (currently focusing on vowels) by running the following script. This will open your camera to capture images for training the model:

```bash
python collect_imgs.py
```

### Dataset Creation

After collecting images, run the script to process and create a dataset for training:

```bash
python create_dataset.py
```

### Model Training

Once the dataset is prepared, you can train the classifier using:

```bash
python train_classifier.py
```

### Real-Time Inference

To use the trained model for real-time sign language detection, run:

```bash
python inference_classifier.py
```

## 🤝 Contributions

This project is a collaborative effort between students and faculty members. Contributions, suggestions, and improvements are welcome. Please feel free to reach out to us.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.