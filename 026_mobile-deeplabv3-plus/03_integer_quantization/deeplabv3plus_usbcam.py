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

# Deeplab color palettes
DEEPLAB_PALETTE = Image.open("colorpalette.png").getpalette()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep_model", default="deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite", help="Path of the deeplabv3plus model.")
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

    interpreter = Interpreter(model_path=deep_model)
    try:
        interpreter.set_num_threads(num_threads)
    except:
        print("WARNING: The installed PythonAPI of Tensorflow/Tensorflow Lite runtime does not support Multi-Thread processing.")
        print("WARNING: It works in single thread mode.")
        print("WARNING: If you want to use Multi-Thread to improve performance on aarch64/armv7l platforms, please refer to one of the below to implement a customized Tensorflow/Tensorflow Lite runtime.")
        print("https://github.com/PINTO0309/Tensorflow-bin.git")
        print("https://github.com/PINTO0309/TensorflowLite-bin.git")
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    deeplabv3_predictions = interpreter.get_output_details()[0]['index']

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
        prepimg_deep = cv2.resize(color_image, (256, 256))
        prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
        prepimg_deep = prepimg_deep.astype(np.float32)
        cv2.normalize(prepimg_deep, prepimg_deep, -1, 1, cv2.NORM_MINMAX)

        # Run model - DeeplabV3-plus
        interpreter.set_tensor(input_details, prepimg_deep)
        interpreter.invoke()

        # Get results
        predictions = interpreter.get_tensor(deeplabv3_predictions)[0]

        # Segmentation
        outputimg = np.uint8(predictions)
        outputimg = cv2.resize(outputimg, (camera_width, camera_height))
        outputimg = Image.fromarray(outputimg, mode="P")
        outputimg.putpalette(DEEPLAB_PALETTE)
        outputimg = outputimg.convert("RGB")
        outputimg = np.asarray(outputimg)
        outputimg = cv2.cvtColor(outputimg, cv2.COLOR_RGB2BGR)
        imdraw = cv2.addWeighted(color_image, 1.0, outputimg, 0.9, 0)

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
