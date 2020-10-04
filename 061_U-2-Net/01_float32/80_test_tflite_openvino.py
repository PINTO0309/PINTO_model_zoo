import os
import cv2
import numpy as np
import time
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IECore

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

def resize_and_pad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h
    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    # set pad color
    if len(img.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_color = [pad_color]*3
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return scaled_img


if __name__ == '__main__':

    camera_index  = 0
    camera_width  = 640
    camera_height = 480

    model_scale = 320

    # Tensorflow Lite
    interpreter = Interpreter(model_path='u2netp_{}x{}_float32.tflite'.format(model_scale, model_scale), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()[0]['index']

    # OpenVINO
    model_xml = 'openvino/{}x{}/FP32/u2netp_{}x{}.xml'.format(model_scale, model_scale, model_scale, model_scale)
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = ie.load_network(network=net, device_name='CPU')

    # Init Camera
    cam = cv2.VideoCapture(camera_index)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    window_name = "USB Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        start_time = time.perf_counter()

        ret, raw_image = cam.read()
        if not ret:
            continue

        image = resize_and_pad(raw_image, (model_scale, model_scale))
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = image.transpose((0, 3, 1, 2))
        image = image / 127.5 - 1.0
        output = exec_net.infer(inputs={input_blob: image})
        output = (output['sigmd0'] + 1) * 127.5
        output = output.transpose((0, 2, 3, 1))[0]
        output = output.astype(np.uint8)
        output = resize_and_pad(output, (camera_width, camera_width))
        cv2.putText(output, detectfps, (camera_width - 175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('USB Camera', output)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        detectframecount += 1
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