# prepare_data.py
import os
import cv2
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
# Tumhare screenshot ke hisaab se paths:
image_folder = "raw_data/images"       # Abhi jo folder rename karwaya
xml_folder = "raw_data/annotations"
output_folder = "dataset"              # Yeh naya folder khud banega

# Classes mapping
class_mapping = {
    "with_mask": "with_mask",
    "without_mask": "without_mask",
    "mask_weared_incorrect": "without_mask" 
}

if not os.path.exists(output_folder):
    os.makedirs(os.path.join(output_folder, "with_mask"))
    os.makedirs(os.path.join(output_folder, "without_mask"))

print("[INFO] Processing start ho rahi hai...")
count = 0

# XML files read karna
if os.path.exists(xml_folder):
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        
        # Image file dhundna
        filename = root.find("filename").text
        base_name = os.path.splitext(xml_file)[0]
        
        # Pehle PNG check karo, phir JPG
        image_path = os.path.join(image_folder, base_name + ".png")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder, base_name + ".jpg")
        
        if not os.path.exists(image_path):
            print(f"[SKIP] Image nahi mili: {base_name}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        # Faces crop karna
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label in class_mapping:
                target_label = class_mapping[label]
                bndbox = obj.find("bndbox")
                
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Boundary checks
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(image.shape[1], xmax), min(image.shape[0], ymax)

                face = image[ymin:ymax, xmin:xmax]
                if face.size > 0:
                    save_path = os.path.join(output_folder, target_label, f"{base_name}_{count}.jpg")
                    cv2.imwrite(save_path, face)
                    count += 1
else:
    print(f"[ERROR] '{xml_folder}' folder nahi mila! Kya tumne folder rename kiya?")

print(f"[INFO] Done! Total {count} faces extracted into 'dataset' folder.")