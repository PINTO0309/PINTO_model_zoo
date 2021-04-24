import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis
import os
import argparse
from yolo_v3.yolo_postprocess import YOLO
from PIL import Image


def process_detection( model, img, bbox, args ):

    y_min, x_min, y_max, x_max = bbox
    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img.shape[1])

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )

    if args.display == 'full':
        cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    return img



def main(args):
    import tensorflow as tf
    whenet = WHENet(snapshot=args.snapshot)
    whenet.model.save('model.h5')
    with tf.keras.backend.get_session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.saved_model.simple_save(sess, 'saved_model', inputs={'input': whenet.model.input}, outputs={t.name:t for t in whenet.model.outputs})
    import sys
    sys.exit(0)

    yolo = YOLO(**vars(args))
    VIDEO_SRC = 0 if args.video == '' else args.video # if video clip is passed, use web cam
    cap = cv2.VideoCapture(VIDEO_SRC)
    print('cap info',VIDEO_SRC)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]))  # write the result to a video

    while True:
        try:
            ret, frame = cap.read()
        except:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        bboxes, scores, classes = yolo.detect(img_pil)
        for bbox in bboxes:
            frame = process_detection(whenet, frame, bbox, args)
        cv2.imshow('output',frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='whenet demo with yolo')
    parser.add_argument('--video', type=str, default='IMG_0176.mp4',         help='path to video file. use camera if no file is given')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
    parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='yolo iou threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--output', type=str, default='test.avi', help='output video name')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)