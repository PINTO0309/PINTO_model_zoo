from PIL import Image
import os, glob
import numpy as np
 
dataset = []

files = glob.glob("*.JPG")
for file in files:
    image = Image.open(file)
    image = image.convert("RGB")
    data = np.asarray(image)
    dataset.append(data)
 
dataset = np.array(dataset)
np.save("janken_dataset", dataset)
