import os
import random

from deepface import DeepFace

files = os.listdir("./img/celebs/")
results = []

def find_acc_of_model():
    for i in range(1000):
        img1_idx = random.randint(0, len(files)-1)
        img2_idx = random.randint(0, len(files)-1)

        if img1_idx == img2_idx:
            continue

        img1 = f"./img/celebs/{files[img1_idx]}"
        img2 = f"./img/celebs/{files[img2_idx]}"

        result = DeepFace.verify(img1_path = img1, img2_path = img2, enforce_detection=False)
        results.append(result["verified"])

        if i % 100 == 0:
            print(f"Epoch {i+1} Done!!")

    print(sum(results)/ len(results))
