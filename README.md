# Verifoolcation
Evading state of the art face verification system such as VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib.

# Steps to Achieve the goal
1. [] Find test accuracy on the base pre-trained models
2. [x] Take a library of faces in a folder
3. [x] Find embedding of all the images and store in a database/ pickle file (use represent function from DeepFace, note to use the "model" parameter
in it to avoid loading)
4. [] Find closest embeddings and make pairs.
5. [] Perform adversarial attack on the same.