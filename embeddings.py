import torch
import pickle
import pathlib as pl

from typing import Dict
from deepface import DeepFace

BASE_PICKLE = "/home/sweta/scratch/verifoolcation/pickles/"

def save_embeddings(img_paths, model_names=["VGG-Face"], models=None, filepath=f"{BASE_PICKLE}non_celebs_embeddings.pkl"):

    result = {}
    for model_name, model in zip(model_names, models):
        # print(model_name, model)
        result[model_name] = []
        for img in img_paths:
            embedding = DeepFace.represent(str(img), model_name=model_name, model=model, enforce_detection=False)
            result[model_name].append([
                str(img), embedding
            ])
        
        print(f"{model_name} Done!!!!")

    with open(filepath, "wb") as f:
        pickle.dump(result, f)


def load_embeddings(filepath=f"{BASE_PICKLE}non_celebs_embeddings.pkl"):
    with open(filepath, "rb") as f:
        embeddings = pickle.load(f)
    
    return embeddings

def compare_embeddings(query_embeddings, embeddings: Dict):
    """
    compare a query embedding with a list of embeddings and
    return minimum distance and the image id.
    """
    result = {}
    for (model, embedds), (query_model, query_embedds) in zip(embeddings.items(), query_embeddings.items()):
        assert model == query_model, "Models mismatch"
        result[model] = []

        for query_embedd in query_embedds:
            image_id = None
            min_distance = float("inf")
            for embedd in embedds:
                stored_embedding = torch.Tensor(embedd[1]).reshape(1, -1)
                query_img = query_embedd[0]
                query_embed = torch.Tensor(query_embedd[1]).reshape(1, -1)
                l2 = torch.cdist(query_embed, stored_embedding)
                if l2 < min_distance:
                    min_distance = l2
                    image_id = embedd[0]
            
            result[model].append([query_img, image_id, min_distance])
        print(f"{model} Done!!")
    
    return result


if __name__ == "__main__":
    update = False
    img_paths = pl.Path("./img/non_celebs").glob("*.*")
    query_image_paths = pl.Path("./img/celebs").glob("*.*")
    model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace"]

    if update:
        models = []
        for model_name in model_names:
            models.append(DeepFace.build_model(model_name))
        save_embeddings(list(img_paths), model_names=model_names, models=models)
        save_embeddings(
            list(query_image_paths), model_names=model_names,\
            models=models, filepath=f"{BASE_PICKLE}celebs_embedding.pkl"
        )
    embeddings: Dict = load_embeddings()
    query_embeddings: Dict = load_embeddings(filepath=f"{BASE_PICKLE}celebs_embedding.pkl")

    result = compare_embeddings(query_embeddings, embeddings)
    print(result)
    # for qip in query_image_paths:
    #     #TODO start with making the query embeddings a dictionary
    #     query_embedding = DeepFace.represent(str(qip), model_name="VGG-Face")
    #     query_embedding = torch.Tensor(query_embedding).reshape(1, -1)
    #     min_dist = compare_embedding(query_embedding, result)
    #     print(qip, min_dist)
    #     break