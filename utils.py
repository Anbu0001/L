import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_face(face_img, target_size=(160, 160)):
    """
    Module 2: Resizing and Normalizing the face for the CNN.
    """
    # Resize to the input size the model expects
    face = cv2.resize(face_img, target_size)
    # Convert to array and scale pixels to [0, 1]
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

def get_liveness_score(face_roi):
    """
    Module 4: Simple Texture Analysis (Laplacian Variance).
    Real faces usually have smoother gradients than printed photos.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Low variance often indicates a blurry 'photo attack' 
    return laplacian_var