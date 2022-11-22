"""
Generate and store facenet embeddings
"""
import os
import torch
import pickle
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

base_img_dir = "img/celebs/"
# base_img_dir = "img_large/"
base_compare_dir = "img/non_celebs/"
images = [base_img_dir+x for x in os.listdir(base_img_dir)]

# Face detection pipeline using MTCNN
mtcnn = MTCNN()

# Create an inception resnet (in eval mode)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def create_embedding(filepath):
    """
    Calculate and store embeddings of celebrities
    (just needed to be run once)
    """
    embeddings = []
    count = 0
    for image in images:
        img = Image.open(image)
    
        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img)

        if img_cropped is not None:

            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = resnet(img_cropped.unsqueeze(0))
            embeddings.append([
                image, img_embedding
            ])
        count += 1
        if count % 100 == 0:
            print(f"{count} images done!!!")

    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)


def compare_embedding(imgpath, filepath):
    """
    Read all embedding and compare
    """
    img = Image.open(imgpath)
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    image_id = None
    min_distance = float("inf")
    for item in obj:
        stored_embedding = item[1]
        l2 = torch.cdist(img_embedding, stored_embedding)
        if l2 < min_distance:
            min_distance = l2
            image_id = item[0]
    
    return image_id, min_distance

# create_embedding("embedding.pkl")
final_images = []
images = [base_compare_dir+x for x in os.listdir(base_compare_dir)]
for image in images:
    image_id, min_distance = compare_embedding(imgpath=image, filepath="embedding.pkl")
    final_images.append((image, image_id))

with open("matching.pkl", "wb") as f:
    pickle.dump(final_images, f)

# print(final_images)
# print(compare_embedding(imgpath="./img/non_celebs/demo.jpg", filepath="embedding.pkl"))
