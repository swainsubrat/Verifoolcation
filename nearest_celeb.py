import os
import pickle
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from deepface import DeepFace

files = os.listdir("./img/celebs/")
results = []

def find_acc_of_model():
    for i in range(1000):
        img1_idx = random.randint(0, len(files)-1)
        img2_idx = random.randint(0, len(files)-1)

        if img1_idx == img2_idx:
            continue

        # print(img1_idx, img2_idx)
        img1 = f"./img/celebs/{files[img1_idx]}"
        img2 = f"./img/celebs/{files[img2_idx]}"

        result = DeepFace.verify(img1_path = img1, img2_path = img2, enforce_detection=False)
        results.append(result["verified"])

        if i % 100 == 0:
            print(f"Epoch {i+1} Done!!")

    print(sum(results)/ len(results))

def plot_image(img1, img2, filename):
    img1 = mpimg.imread(img1)
    img2 = mpimg.imread(img2)
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(img1)
    axis[1].imshow(img2)
    plt.savefig(filename, dpi=600)


def attack(pickle_file_path: str):
    with open(pickle_file_path, "rb") as f:
        image_pairs = pickle.load(f)

    for image_pair in image_pairs:
        result = DeepFace.verify(img1_path = image_pair[0], img2_path = image_pair[1], enforce_detection=False)
        if result["verified"]:
            plot_image(image_pair[0], image_pair[1], f"./plots/img_{random.randint(0, 100000)}.jpg")

attack(pickle_file_path="matching.pkl")