import cv2
import os
import csv

folder = r"C:\Users\KIIT0001\Desktop\API\Image"

files = [f for f in os.listdir(folder)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]

with open("image_info.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["File Name", "Width", "Height", "Channels"])

    for img_name in files:
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)

        if img is not None:
            h, w, c = img.shape
            writer.writerow([img_name, w, h, c])
