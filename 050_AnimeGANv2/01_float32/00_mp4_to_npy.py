import cv2
import numpy as np

def readClip(filepass):
    cap = cv2.VideoCapture(filepass)
    print(cap.isOpened())
    ret,frame = cap.read()
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    array = [np.reshape(frame, (256, 256, 3))]
    frame_count = 0
    while True: 
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (256, 256, 3))
        array = np.append(array, [frame], axis=0)
        frame_count += 1
        if frame_count >= 100:
            break
    cap.release()
    return array

dataset = readClip("video/output/Hayao/お花見.mp4")
np.save("animeganv2_dataset_hayao_256x256", dataset)

dataset = readClip("video/output/Paprika/お花見.mp4")
np.save("animeganv2_dataset_paprika_256x256", dataset)
