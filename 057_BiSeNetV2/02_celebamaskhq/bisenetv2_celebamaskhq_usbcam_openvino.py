import sys
import argparse
import numpy as np
import cv2
import time
import sys
import os

try:
    from armv7l.openvino.inference_engine import IENetwork, IECore
except:
    from openvino.inference_engine import IENetwork, IECore

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
    # mask_color = np.zeros(shape=[3, mask_shape[1], mask_shape[2]], dtype=np.uint8)
    unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]
    for label_id in unique_label_ids:
        idx = np.where(mask == label_id)
        mask_color[idx] = LABEL_CONTOURS[label_id]
    return mask_color


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep_model", default="openvino/bisenetv2_celebamaskhq_448x448/FP16/bisenetv2_celebamaskhq_448x448.xml",
                                        help="Path of the BiSeNetV2 model. Integer Quant models are optimized for ARM. Try Weight Quant if you are using x86.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument('--camera_width', type=int, default=640, help='USB Camera resolution (width). (Default=640)')
    parser.add_argument('--camera_height', type=int, default=480, help='USB Camera resolution (height). (Default=480)')
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')
    parser.add_argument('--device', type=str, default='CPU', help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                                   Sample will look for a suitable plugin for device specified (CPU by default)')
    args = parser.parse_args()

    deep_model    = args.deep_model
    usbcamno      = args.usbcamno
    vidfps        = args.vidfps
    camera_width  = args.camera_width
    camera_height = args.camera_height
    device        = args.device

    model_xml = deep_model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model_xml, model_bin)
    input_info = net.input_info
    _, _, input_height, input_width = net.inputs['input_tensor'].shape
    input_blob = next(iter(input_info))
    exec_net = ie.load_network(network=net, device_name=device)

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
        prepimg_deep = cv2.resize(color_image, (input_width, input_height))
        prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
        prepimg_deep = prepimg_deep.astype(np.float32)
        prepimg_deep /= 255.0
        prepimg_deep -= [[[0.5, 0.5, 0.5]]]
        prepimg_deep /= [[[0.5, 0.5, 0.5]]]
        prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])

        # Run model
        predictions = exec_net.infer(inputs={input_blob: prepimg_deep})

        # Segmentation
        imdraw = decode_prediction_mask(predictions['final_output'])
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
