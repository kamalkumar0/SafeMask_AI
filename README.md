# 🛡️ SafeMask AI — Smart Face Mask Detection System

SafeMask AI is a real-time face mask detection system built using Deep Learning and Computer Vision.  
The project uses a trained MobileNetV2 model to detect whether a person is wearing a mask or not through a live webcam feed.

The main idea behind this project was to create a practical AI-based monitoring solution that can help improve safety compliance in public environments such as offices, campuses, hospitals, and workplaces.

This project combines Machine Learning, Flask backend, and a modern frontend dashboard to create a complete working system.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Running the Application](#running-the-application)
- [Screen Recording](#screen-recording)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🚀 Project Overview

The system detects faces from a webcam stream and classifies them into two categories:

- With Mask
- Without Mask

A deep learning model trained on face images performs classification, while OpenCV handles real-time face detection and visualization.

The frontend dashboard provides a clean interface showing:

- Live camera feed
- Detection status
- Accuracy information
- System controls

---

## ✨ Features

- Real-time face mask detection using webcam
- Deep learning classification with MobileNetV2
- Modern dashboard UI
- Live monitoring status
- Detection bounding boxes with confidence scores
- Lightweight and fast performance
- Easy to run locally
- Modular code structure (training + app separation)

---

## 🧠 System Architecture

1. Dataset images are processed and faces are extracted
2. Deep learning model is trained using MobileNetV2
3. Model is saved as `.h5` file
4. Flask server loads the trained model
5. Webcam frames are captured using OpenCV
6. Faces are detected and classified in real time
7. Results are streamed to the frontend dashboard

---

## 🔄 How It Works

When the application starts:

- Flask initializes the server
- Face detection model (Haar Cascade) loads
- Trained mask classifier loads
- Webcam frames are captured continuously
- Each detected face is preprocessed and passed to the model
- Prediction results are displayed with bounding boxes

The training pipeline is implemented in the model script :contentReference[oaicite:0]{index=0} which prepares the dataset, applies augmentation, trains MobileNetV2, and saves the model.

---

## 🛠 Tech Stack

### Machine Learning
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Scikit-learn

### Computer Vision
- OpenCV

### Backend
- Python
- Flask

### Frontend
- HTML
- CSS
- JavaScript
- Bootstrap

---

## 📂 Dataset Preparation

Before training the model, the raw dataset is processed using annotation files.

Steps include:

- Reading XML annotation files
- Extracting face regions
- Mapping labels into two categories
- Saving cropped faces into structured folders

Output folder structure:
