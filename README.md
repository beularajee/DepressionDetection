EEG-Based Depression Detection using Deep Learning 
This project presents an intelligent system that uses EEG (Electroencephalogram) images to predict signs of depression using deep learning. The solution integrates a trained Convolutional Neural Network (CNN) model with an interactive Streamlit web application, enabling users to upload EEG images and receive real-time predictions.
This project was developed as part of a practical AI/ML application in healthcare and can be expanded into clinical research, mental health diagnostics, or telehealth platforms.

PROJECT HIGHLIGHTS:

->Medical AI: Targets neurological disorders using EEG signals.
->Image-Based Classification: Works on EEG image scans (converted signals).
->Deep Learning: Trained CNN model with high accuracy.
->Web App: Interactive user interface using Streamlit.
->Dataset Source: Custom dataset hosted on Roboflow with Positive/Negative EEG samples.

PROBLEM STATEMENT:

Mental health disorders, especially depression, are a growing concern worldwide. Traditional diagnosis relies heavily on subjective assessments. This project explores the automated detection of depression by analyzing EEG images, enabling faster, scalable, and more objective diagnosis support.

MODEL OVERVIEW:

| Feature        | Value                                       |
| -------------- | ------------------------------------------- |
| Model Type     | Convolutional Neural Network (CNN)          |
| Input          | EEG Images (224x224x3 RGB)                  |
| Output         | Binary Classification (Positive / Negative) |
| Activation     | ReLU (hidden layers), Sigmoid (output)      |
| Loss           | Binary Crossentropy                         |
| Optimizer      | Adam                                        |
| Framework      | TensorFlow / Keras                          |

DATASET:

Name: EEG_DataSet_P_N
Source: Roboflow Dataset
Format: Image (.png/.jpg)
Classes: Positive, Negative
Images are labeled and organized using a .csv file and classified during preprocessing using a custom script.
DEMO (STREAMLIT APP):
<p align="center"> <img src="https://i.imgur.com/w8JwjXJ.png" width="600" alt="Streamlit EEG Depression App Preview"> </p>

PROJECT STRUCTURE:

├── app.py                        # Streamlit web interface
├── depression_detection_model.h5 # Trained CNN model
├── preprocess.py                 # CSV-based file sorting script
├── dataset/
│   ├── positive/                 # Positive EEG images
│   └── negative/                 # Negative EEG images
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

HOW IT WORKS:

*User uploads an EEG image.
*Image is resized and normalized.
*Preprocessed image is passed to a trained CNN model.
*Model predicts whether depression is Positive or Negative.
*Result is displayed on the web interface.

FEATURES:

*Real-time prediction using EEG images.
*Clean UI built with Streamlit.
*Easily extendable for clinical use or research.
*Portable and fully open-source.

TECHNOLOGIES USED:

*Python
*TensorFlow / Keras
*Streamlit
*NumPy, Pandas, Pillow
*Roboflow (for dataset management)








