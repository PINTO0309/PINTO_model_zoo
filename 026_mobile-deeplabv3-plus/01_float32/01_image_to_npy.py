from PIL import Image
import os, glob
import numpy as np
 
dataset = []

files = glob.glob("JPEGImages/*.jpg")
for file in files:
    image = Image.open(file)
    image = image.convert("RGB")
    data = np.asarray(image)
    dataset.append(data)
 
dataset = np.array(dataset)
np.save("person_dataset", dataset)
