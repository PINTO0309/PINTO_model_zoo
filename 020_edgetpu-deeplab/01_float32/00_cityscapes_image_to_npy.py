from PIL import Image
import os, glob
import numpy as np
 
dataset = []

files = glob.glob("bielefeld/*.png")
for file in files:
    image = Image.open(file)
    data = np.asarray(image)
    dataset.append(data)
 
dataset = np.array(dataset)
np.save("cityscapes_bielefeld_181", dataset)
