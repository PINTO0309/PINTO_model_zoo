import cv2
import numpy as np
import sys

#path = "/media/b920405/Windows/datasets/VOC/train/VOCdevkit/VOC2012/JPEGImages/" ##pic path
path = []
name = []
bbox = []
with open("./voc_train.txt","r") as vocfile: ##change your file‘s path
    for lines in vocfile.readlines():
        line = lines.split(' ')
        path.append(line[0])
        name.append(line[0].split('/')[-1])
        bbox.append(np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]]))

    wrongnames = []
    for i in range(len(name)):
        print("Exec:", path[i])
        image = cv2.imread(path[i])
        shape = image.shape
        for box in bbox[i]:
            if box[0] < 0 or box[0] > shape[1] or box[2] < 0 or box[2] > shape[1] or box[1] < 0 or box[1] > shape[0] or box[3] < 0 or box[3] > shape[0] or box[4] != 0:
                wrongnames.append(name[i])
                print(name[i] + " wrong:", "b0", box[0], "b1", box[1], "b2", box[2], "b3", box[3], "b4", box[4], shape[0], shape[1])
            else:
                print(str(i) + " right")
                print("done")
        sys.exit(0)

print(wrongnames)