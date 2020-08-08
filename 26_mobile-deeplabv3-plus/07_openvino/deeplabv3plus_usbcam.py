import sys
import os
import argparse
import numpy as np
import cv2
import time
from PIL import Image

try:
    from armv7l.openvino.inference_engine import IENetwork, IECore
except:
    from openvino.inference_engine import IENetwork, IECore

fps = ""
framecount = 0
time1 = 0

# Deeplab color palettes
DEEPLAB_PALETTE = Image.open("colorpalette.png").getpalette()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep_model", default="openvino/256x256/FP16/deeplab_v3_plus_mnv3_decoder_256.xml", help="Path of the deeplabv3plus model.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument('--camera_width', type=int, default=640, help='USB Camera resolution (width). (Default=640)')
    parser.add_argument('--camera_height', type=int, default=480, help='USB Camera resolution (height). (Default=480)')
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')
    parser.add_argument('--device', type=str, default='CPU', help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                                   Sample will look for a suitable plugin for device specified (CPU by default)')
    args = parser.parse_args()
    deep_model    = args.deep_model
    usbcamno      = args.usbcamno
    camera_width  = args.camera_width
    camera_height = args.camera_height
    vidfps        = args.vidfps
    device        = args.device

    model_xml = deep_model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    ie = IECore()
    net = ie.read_network(model_xml, model_bin)
    input_info = net.input_info
    input_blob = next(iter(input_info))
    exec_net = ie.load_network(network=net, device_name=args.device)

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
        prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])
        prepimg_deep -= 127.5
        prepimg_deep /= 127.5

        # Run model - DeeplabV3-plus
        deeplabv3_predictions = exec_net.infer(inputs={input_blob: prepimg_deep})

        # Get results
        predictions = deeplabv3_predictions['Output/Transpose']

        # Segmentation
        outputimg = np.uint8(predictions[0][0])
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
