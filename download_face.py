# download_face.py
import requests
import os

# Folder create karo agar nahi hai
if not os.path.exists('face_detector'):
    os.makedirs('face_detector')

files = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

print("[INFO] Downloading face detector files... Please wait.")
for name, url in files.items():
    path = f"face_detector/{name}"
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        try:
            r = requests.get(url)
            with open(path, 'wb') as f:
                f.write(r.content)
            print(f"✔ {name} saved.")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
    else:
        print(f"⚠ {name} pehle se मौजूद hai.")

print("[SUCCESS] Files ready! Ab app.py run karo.")
