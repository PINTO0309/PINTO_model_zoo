
import common
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from argparse import ArgumentParser
from tensorflow.lite.python.interpreter import Interpreter


def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(interpreter, input_blob, output_blob, image, threshold=0.4, nms_iou=0.5):

    interpreter.set_tensor(input_blob[0]['index'], image)
    interpreter.invoke()

    # hm, box, landmark = outputs['1028'], outputs['1029'], outputs['1027']
    lm = interpreter.get_tensor(output_blob[0]['index']).transpose((0,3,1,2)) # 1,h,w,10
    box = interpreter.get_tensor(output_blob[1]['index']).transpose((0,3,1,2)) # 1,h,w,4
    hm = interpreter.get_tensor(output_blob[2]['index']).transpose((0,3,1,2)) # 1,1,h,w

    x = torch.from_numpy(hm).clone()
    y = torch.from_numpy(box).clone()
    z = torch.from_numpy(lm).clone()
    for var in [x, y, z]:
        if var.shape[1]==1:
            hm = var
        elif var.shape[1]==4:
            box = var
        elif var.shape[1]==10:
            landmark = var
    
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def camera_demo():
    interpreter = Interpreter(model_path='model_float32.tflite', num_threads=4)
    interpreter.allocate_tensors()
    input_blob = interpreter.get_input_details()
    output_blob = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        img = cv2.resize(frame, (640,480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]
        objs = detect(interpreter, input_blob, output_blob, img)

        for obj in objs:
            common.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_demo()
