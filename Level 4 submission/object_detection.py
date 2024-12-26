import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"

model = YOLO(MODEL_NAME)

img1_path = "level-4/datasets/awmleer-I--YyrXUphc-unsplash.jpg"
img2_path = "level-4/datasets/daniel-lloyd-blunk-fernandez-QkKKggRWlE8-unsplash.jpg"
img3_path = "level-4/datasets/gabriel-martin-FH3NzSqwOTU-unsplash.jpg"

img_path = [img1_path,img2_path,img3_path]

for _ in range(len(img_path)):
    results = model(img_path[_])

    # Plot and log the results
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
    ax.axis("off")

    plt.show()
