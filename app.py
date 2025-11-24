from flask import Flask, render_template, Response
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# --- 1. LOAD MODELS ---
print("-----------------------------------")
print("[INFO] Models loading process started...")

# Method 1: Built-in Face Detector (No external files needed)
# Yeh OpenCV ke andar pehle se hota hai
face_detector_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceNet = cv2.CascadeClassifier(face_detector_path)

if faceNet.empty():
    print("[ERROR] Built-in Face Detector nahi mila! OpenCV reinstall karna pad sakta hai.")
else:
    print("[SUCCESS] Built-in Face Detector Loaded!")

# Mask Model (Jo tumne train kiya)
try:
    maskNet = load_model("mask_detector.h5")
    print("[SUCCESS] Mask Detector Loaded successfully!")
except Exception as e:
    print(f"[CRITICAL ERROR] Mask Model load nahi hua! Reason: {e}")
print("-----------------------------------")


def detect_and_predict_mask(frame, faceNet, maskNet):
    # Haar Cascade ko Gray image chahiye hoti hai
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    # Face Detection (Scale Factor=1.1, Min Neighbors=4)
    faces = faceNet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    faces_list = []
    locs = []
    preds = []

    for (x, y, w_box, h_box) in faces:
        # Box ke coordinates set karo
        startX, startY, endX, endY = x, y, x + w_box, y + h_box
        
        # Frame se chehra kato
        face = frame[startY:endY, startX:endX]
        
        if face.any():
            # Image preprocessing (Model ke hisaab se)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces_list.append(face)
            locs.append((startX, startY, endX, endY))

    # Agar chehra mila, toh Mask check karo
    if len(faces_list) > 0:
        faces_list = np.array(faces_list, dtype="float32")
        preds = maskNet.predict(faces_list, batch_size=32)
    
    return (locs, preds)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Webcam open nahi ho raha!")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Frame resize (Taaki tez chale)
        frame = cv2.resize(frame, (800, 600))
        frame = cv2.flip(frame, 1) # Mirror effect

        # Detection Call
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        # Loop over faces
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            # Logic: Mask vs No Mask
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # Probability show karo
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Box aur Text draw karo
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)