#This file will hold whatever implementation of clip that we want to use.
#Hopefully it will gather information about the image and output a vector
#of word descriptions


#Just for general environment

#Have to do pip install transformers

#Code below from https://www.kaggle.com/code/batprem/image-classification-without-training

# Import transformer models

from transformers import (
    CLIPProcessor,
    CLIPModel
)


# Reqeust
import requests

# Image object
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dog_images = [
    'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/golden-retriever-royalty-free-image-506756303-1560962726.jpg?crop=1.00xw:0.756xh;0,0.0756xh&resize=980:*',
    'https://cdn.pixabay.com/photo/2020/06/02/03/18/siberian-husky-5249166_1280.jpg',
    'https://p1.pxfuel.com/preview/161/627/771/corgi-sitting-waiting-looking-dog-autumn.jpg'
]

cat_images = [
    'https://upload.wikimedia.org/wikipedia/commons/c/c7/Tabby_cat_with_blue_eyes-3336579.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/b/b1/VAN_CAT.png',
    'https://cdn.pixabay.com/photo/2019/10/04/20/55/white-cat-4526549_1280.jpg'
]

image = Image.open(requests.get(dog_images[2], stream=True).raw)

texts = [
    "a photo of a cat",
    "a photo of a dog"
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)

print(dict(zip(texts, probs[0].detach().numpy())))
image

image = Image.open(requests.get(cat_images[2], stream=True).raw)

texts = [
    "a photo of a cat",
    "a photo of a dog"
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)

print(dict(zip(texts, probs[0].detach().numpy())))
image


image = Image.open(requests.get(dog_images[0], stream=True).raw)

texts = [
    "a photo of a short-legs dog",
    "a photo of a husky",
    "a photo of a golden dog"
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)

print(dict(zip(texts, probs[0].detach().numpy())))
image

image = Image.open(requests.get(dog_images[1], stream=True).raw)

texts = [
    "a photo of a short-legs dog",
    "a photo of a husky",
    "a photo of a golden dog"
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)

print(dict(zip(texts, probs[0].detach().numpy())))
image





