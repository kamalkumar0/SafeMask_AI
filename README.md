# 🛡️ SafeMask AI — Smart Face Mask Detection System

SafeMask AI is a real-time face mask detection system built using Deep Learning and Computer Vision.
The project uses a trained MobileNetV2 model to detect whether a person is wearing a mask or not through a live webcam feed.

The main idea behind building this project was to create a practical AI-based monitoring solution that can help improve safety compliance in environments like offices, campuses, hospitals, and workplaces.

This project combines Machine Learning, a Flask backend, and a modern frontend dashboard to create a complete working AI system.

## 📑 Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [System Architecture](#system-architecture)
* [How It Works](#how-it-works)
* [Tech Stack](#tech-stack)
* [Dataset Preparation](#dataset-preparation)
* [Model Training](#model-training)
* [Running the Application](#running-the-application)
* [Screen Recording](#screen-recording)
* [Future Improvements](#future-improvements)
* [Author](#author)

<a id="project-overview"></a>

## 🚀 Project Overview

The system detects faces from a webcam stream and classifies them into two categories:

* With Mask
* Without Mask

A deep learning model trained on face images performs classification, while OpenCV handles real-time face detection and visualization.

The frontend dashboard provides a clean interface showing:

* Live camera feed
* Detection status
* Accuracy information
* System controls

<a id="features"></a>

## ✨ Features

* Real-time face mask detection using webcam
* Deep learning classification with MobileNetV2
* Modern dashboard UI
* Live monitoring status
* Detection bounding boxes with confidence scores
* Lightweight and fast performance
* Easy to run locally
* Modular code structure (training + deployment separation)

<a id="system-architecture"></a>

## 🧠 System Architecture

1. Dataset images are processed and faces are extracted
2. Deep learning model is trained using MobileNetV2
3. Model is saved as `.h5` file
4. Flask server loads the trained model
5. Webcam frames are captured using OpenCV
6. Faces are detected and classified in real time
7. Results are streamed to the frontend dashboard

<a id="how-it-works"></a>

## 🔄 How It Works

When the application starts:

* Flask initializes the server
* Face detection model loads
* Trained mask classifier loads
* Webcam frames are captured continuously
* Each detected face is preprocessed and passed to the model
* Prediction results are displayed with bounding boxes

The training pipeline prepares the dataset, applies augmentation, trains MobileNetV2, and saves the final model for deployment.

<a id="tech-stack"></a>

## 🛠 Tech Stack

### Machine Learning

* TensorFlow / Keras
* MobileNetV2
* NumPy
* Scikit-learn

### Computer Vision

* OpenCV

### Backend

* Python
* Flask

### Frontend

* HTML
* CSS
* JavaScript
* Bootstrap

<a id="dataset-preparation"></a>

## 📂 Dataset Preparation

The dataset used for training was collected from Kaggle and then preprocessed for this project.

Since the raw dataset contained annotated images, a custom preprocessing script was used to:

* Read XML annotation files
* Extract face regions using bounding box coordinates
* Convert multiple labels into two categories:

  * with_mask
  * without_mask
* Save cropped face images into structured folders for training

Final dataset structure:

```
dataset/
    with_mask/
    without_mask/
```

Dataset Source: Kaggle (Face Mask Detection Dataset)

Note: The dataset was cleaned and reorganized specifically for this project to improve training performance.

<a id="model-training"></a>

## 🤖 Model Training

The model is based on MobileNetV2 transfer learning.

Training process includes:

* Image preprocessing (224x224)
* Data augmentation
* Freezing base layers
* Training custom classification head
* Validation split (80/20)
* Accuracy and loss visualization
* Saving model as `mask_detector.h5`

Transfer learning helps achieve high accuracy even with a limited dataset.

<a id="running-the-application"></a>

## ▶️ Running the Application

### Step 1 — Install Dependencies

```
pip install -r requirements.txt
```

If requirements file is not available:

```
pip install tensorflow flask opencv-python numpy scikit-learn matplotlib
```

### Step 2 — Train Model (Optional)

```
python train_model.py
```

### Step 3 — Run Flask App

```
python app.py
```

### Step 4 — Open Browser

```
http://127.0.0.1:5000
```

<a id="screen-recording"></a>

## 🎥 Screen Recording

You can place your project demo video or GIF here.

Example:

```
Add your screen recording link here
```

<a id="future-improvements"></a>

## 🔮 Future Improvements

Some possible enhancements:

* Multiple camera support
* Alert system for violations
* Database logging
* Email or SMS notifications
* Cloud deployment
* Face recognition integration
* Admin authentication system

<a id="author"></a>

## 👨‍💻 Author

Kamal Kumar
B.Tech CSE Student

🔗 LinkedIn: https://www.linkedin.com/in/kamalkumar0

If you found this project useful, feel free to connect or share feedback.
