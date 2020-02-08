import numpy as np
import math
import time
import sys
import cv2
import argparse
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

LABELS = [
'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="03_integer_quantization/ssdlite_mobilenet_v2_voc_300_integer_quant_with_postprocess.tflite", help="Path of the detection model.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument("--camera_width", type=int, default=320, help="width.")
    parser.add_argument("--camera_height", type=int, default=240, help="height.")
    parser.add_argument("--vidfps", type=int, default=30, help="Frame rate.")
    parser.add_argument("--num_threads", type=int, default=4, help="Threads.")
    args = parser.parse_args()

    model        = args.model
    usbcamno     = args.usbcamno
    image_width  = args.camera_width
    image_height = args.camera_height
    vidfps       = args.vidfps
    num_threads  = args.num_threads

    interpreter = Interpreter(model_path=model)
    try:
        interpreter.set_num_threads(num_threads)
    except:
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cam = cv2.VideoCapture(usbcamno)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    window_name = "USB Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        start_time = time.perf_counter()

        ret, image = cam.read()
        if not ret:
            continue

        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        cv2.normalize(frame, frame, -1, 1, cv2.NORM_MINMAX)

        # run model
        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()

        # get results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = interpreter.get_tensor(output_details[3]['index'])[0]

        # draw boxes
        for i, (box, classidx, score) in enumerate(zip(boxes, classes, scores)):
            probability = score
            if probability >= 0.6:
                ymin = int(box[0] * image_height)
                xmin = int(box[1] * image_width)
                ymax = int(box[2] * image_height)
                xmax = int(box[3] * image_width)
                if ymin > ymax:
                    continue
                if xmin > xmax:
                    continue
                classnum = int(classidx)
                #print('coordinates: ({}, {})-({}, {}). class: "{}". probability: {:.2f}'.format(xmin, ymin, xmax, ymax, classnum, score))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, '{}: {:.2f}'.format(LABELS[classnum], score), (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, fps, (image_width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, detectfps, (image_width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
            if i >= (count - 1):
                break

        cv2.imshow('USB Camera', image)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        detectframecount += 1

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fps = "(Playback) {:.1f} FPS".format(time1 / 10)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount / time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        end_time = time.perf_counter()
        elapsedTime = end_time - start_time
        time1 += 1 / elapsedTime
        time2 += elapsedTime
