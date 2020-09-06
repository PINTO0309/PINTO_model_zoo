import sys
import argparse
import numpy as np
import cv2
import time
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

fps = ""
framecount = 0
time1 = 0

LABEL_CONTOURS = [(0, 0, 0),  # 0=background
                  # 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l
                  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  # 16=cloth, 17=hair, 18=hat
                  (0, 64, 0), (128, 64, 0), (0, 192, 0)]

def decode_prediction_mask(mask):
    mask_shape = mask.shape
    mask_color = np.zeros(shape=[mask_shape[0], mask_shape[1], 3], dtype=np.uint8)
    unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]
    for label_id in unique_label_ids:
        idx = np.where(mask == label_id)
        mask_color[idx] = LABEL_CONTOURS[label_id]
    return mask_color


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep_model", default="bisenetv2_celebamaskhq_448x448_integer_quant.tflite",
                                        help="Path of the BiSeNetV2 model. Integer Quant models are optimized for ARM. Try Weight Quant if you are using x86.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument('--camera_width', type=int, default=640, help='USB Camera resolution (width). (Default=640)')
    parser.add_argument('--camera_height', type=int, default=480, help='USB Camera resolution (height). (Default=480)')
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')
    parser.add_argument("--num_threads", type=int, default=4, help="Threads.")
    args = parser.parse_args()

    deep_model    = args.deep_model
    usbcamno      = args.usbcamno
    vidfps        = args.vidfps
    camera_width  = args.camera_width
    camera_height = args.camera_height
    num_threads   = args.num_threads

    interpreter = Interpreter(model_path=deep_model, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    bisenetv2_predictions = interpreter.get_output_details()[0]['index']
    model_height = interpreter.get_input_details()[0]['shape'][1]
    model_width = interpreter.get_input_details()[0]['shape'][2]

    # print('interpreter.get_output_details()[0]:', interpreter.get_output_details()[0])

    cam = cv2.VideoCapture(usbcamno)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    waittime = 1
    window_name = "USB Camera"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        ret, color_image = cam.read()
        if not ret:
            continue

        # Normalization
        prepimg_deep = cv2.resize(color_image, (model_width, model_height))
        prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
        prepimg_deep = prepimg_deep.astype(np.float32)
        prepimg_deep /= 255.0
        prepimg_deep -= [[[0.5, 0.5, 0.5]]]
        prepimg_deep /= [[[0.5, 0.5, 0.5]]]

        # Run model
        interpreter.set_tensor(input_details, prepimg_deep)
        interpreter.invoke()

        # Get results
        predictions = interpreter.get_tensor(bisenetv2_predictions)

        # Segmentation
        imdraw = decode_prediction_mask(predictions)[:, :, (2, 1, 0)]
        imdraw = cv2.cvtColor(imdraw, cv2.COLOR_RGB2BGR)
        imdraw = cv2.resize(imdraw, (camera_width, camera_height))

        cv2.putText(imdraw, fps, (camera_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, imdraw)

        if cv2.waitKey(waittime)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fps       = "(Playback) {:.1f} FPS".format(time1/10)
            framecount = 0
            time1 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
