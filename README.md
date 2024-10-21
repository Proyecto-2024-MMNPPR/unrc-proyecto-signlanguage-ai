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

To run the project, you need to have Python and Node.js installed on your system. Follow the steps below to set up the project:

In the root of the project, you will find ``setup.sh`` script, which will help you to install the required dependencies, set up, and run the project.

### Install Dependencies
To install the dependencies for the project, run the following command:

```bash
setup.sh dep install
```

### AI
To create dataset, train the model and run the AI, you need to run the following commands:

First, record the samples, the data needed to train the model:
```bash
setup.sh ai capture
```

Then, train the model:
```bash
setup.sh ai train
```

Finally, run the AI to detect the sign language:
```bash
setup.sh ai run
```

## 🤝 Contributions

This project is a collaborative effort between students and faculty members. Contributions, suggestions, and improvements are welcome. Please feel free to reach out to us.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
