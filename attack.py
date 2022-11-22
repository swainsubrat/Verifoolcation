import os
import sys
import pickle
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from deepface import DeepFace

def plot_image(img1, img2, filename):
    img1 = mpimg.imread(img1)
    img2 = mpimg.imread(img2)
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(img1)
    axis[1].imshow(img2)
    plt.savefig(filename, dpi=600)


def attack(pickle_file_path: str, model_name="VGG-Face"):
    with open(pickle_file_path, "rb") as f:
        image_pairs = pickle.load(f)
    
    if not os.path.exists(f"plots/{model_name}"):
        os.makedirs(f"plots/{model_name}")

    for image_pair in image_pairs:
        result = DeepFace.verify(
            img1_path = image_pair[0],
            img2_path = image_pair[1],
            enforce_detection=False,
            model_name=model_name,
            # detector_backend = "ssd", # ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
        )
        image_name = image_pair[1].split("/")[-1]
        if result["verified"]:
            plot_image(image_pair[0], image_pair[1], f"./plots/{model_name}/{image_name}")

if __name__ == "__main__":
    model_name = None
    try:
        # Pass one of these: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace"
        valid_names = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
        model_name = sys.argv[1]
        if model_name not in valid_names:
            print("Entered model name not found!!!!, using default VGG-Face")
            print("Choose from these: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace")
            raise Exception
    except:
        print("In the exception!!!!")
        model_name = "VGG-Face"
    
    # valid_names = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
    valid_names = ["VGG-Face", "Facenet"]
    for model_name in valid_names:
        attack(pickle_file_path="matching.pkl", model_name=model_name)
        print(f"{model_name} Done!!!")