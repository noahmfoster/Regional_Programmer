# import matplotlib.pyplot as plt
import torch as torch
import numpy as np

import cv2
import os
import numpy as np
import pickle


import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras_vggface import utils

from transformers import CLIPProcessor, CLIPModel


import keras.utils as image
import csv
from tensorflow.keras.models import load_model



data_path = os.path.dirname(os.path.abspath(__file__)) + '/../Data/'


# load the training labels
face_label_filename = data_path + 'face-labels.pickle'
with open(face_label_filename, "rb") as \
    f: class_dictionary = pickle.load(f)

class_list = [value for _, value in class_dictionary.items()]
# print(class_list)

# load list of 73 nouns 

def get_nouns():
    with open(data_path + 'nouns.csv', newline='') as f:
        reader = csv.reader(f)
        words = list(reader)

    words = [line[0] for line in words]

    nouns = ['a photo of a ' + s for s in words]

    return nouns, words


def get_models(model_path = None):
    # pretrained clip model and processor for object detection
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print('CLIP model loaded')
    # custom transfer model for detecting characters
    if model_path is None:
        model_path  = os.path.dirname(os.path.abspath(__file__)) +"/"
    face_model = load_model(
        model_path + 'transfer_learning_trained_the_office_cnn_model.h5')

    # pretrained model to detect faces


    facecascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    return clip_model, processor, face_model, facecascade

def run_vision_model(file, clip_model, processor, face_model, facecascade, nouns, words):
    print('running vision model')

    assert file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"), "Image file must be of type .png, .jpg, or .jpeg"

    # load the image
    imgtest = cv2.imread(file, cv2.IMREAD_COLOR)
    image_array = np.array(imgtest, "uint8")
    

    # get the faces detected in the image
    all_faces = facecascade.detectMultiScale(imgtest, 
        scaleFactor=1.1, minNeighbors=5)

    print(f"Looping through {len(all_faces)} faces")
    # loop through each face and get each character
    character_vector = []
    for i in range(len(all_faces)):
        faces = [all_faces[i]]
        for (x_, y_, w, h) in faces:
            # draw the face detected
            face_detect = cv2.rectangle(
                imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)

            # resize the detected face to 224x224
            # size = (image_width, image_height)
            size = (224, 224)
            roi = image_array[y_: y_ + h, x_: x_ + w]
            resized_image = cv2.resize(roi, size)

            # prepare the image for prediction
            x = image.img_to_array(resized_image)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)
            print(x)

            # making prediction
            predicted_prob = face_model.predict(x)
            character_vector += [class_list[predicted_prob[0].argmax()]]
            print('character: ', class_list[predicted_prob[0].argmax()])
            
        inputs = processor(text=nouns, images=imgtest, return_tensors="pt", padding=True)

        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilitie
        #print(probs)
        #ind = torch.topk(probs.flatten(), 3).indices
        ind = probs.squeeze().argsort()[-5:]
        sentence = [words[k] for k in ind]
        plt.imshow(cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB))
        plt.show()
        print('character: ', character_vector)
        print('objects: ', sentence)
        
        return character_vector, sentence # return the character and the top 5 objects detected in the image

