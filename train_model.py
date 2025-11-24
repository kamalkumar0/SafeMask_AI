# train_model.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# --- SETTINGS ---
INIT_LR = 1e-4       # Learning rate
EPOCHS = 20          # Kitni baar training chalegi
BS = 32              # Batch size
DIRECTORY = "dataset"  # Yeh wo folder hai jo prepare_data.py ne banaya hai
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Images load ho rahi hain...")

data = []
labels = []

# 1. Dataset Folder se images read karna
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    if not os.path.exists(path):
        print(f"[ERROR] Folder nahi mila: {path}")
        continue
        
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            # Image load karo aur size 224x224 fix karo
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image) # MobileNetV2 format

            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"[SKIP] Image load nahi hui: {img_path}")

# 2. Labels ko numbers mein badalna (Binary)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# 3. Training aur Testing data alag karna (80% Train, 20% Test)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 4. Data Augmentation (Images ko thoda ghuma-phira kar sikhana)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# 5. Model Setup (MobileNetV2)
print("[INFO] Model ready ho raha hai...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Naya head banana (Classification ke liye)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 2 classes: Mask / No Mask

model = Model(inputs=baseModel.input, outputs=headModel)

# Base model freeze karna (taaki purani learning kharab na ho)
for layer in baseModel.layers:
	layer.trainable = False

# 6. Compile aur Train
print("[INFO] Training start... (Isme 5-10 min lagenge)")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# 7. Report Generate karna
print("[INFO] Testing result...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# 8. Save Model
print("[INFO] Model save kiya ja raha hai as 'mask_detector.h5'...")
model.save("mask_detector.h5")

# 9. Graph Save karna
print("[INFO] Graph save kiya ja raha hai as 'plot.png'...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

print("[SUCCESS] Sab kaam ho gaya! Ab tum app.py run kar sakte ho.")