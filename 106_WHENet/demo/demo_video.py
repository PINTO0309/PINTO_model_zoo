"""
xhost +local: && \
docker run --gpus all -it --rm \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--device /dev/video1:/dev/video1:mwr \
--device /dev/video2:/dev/video2:mwr \
--device /dev/video3:/dev/video3:mwr \
--device /dev/video4:/dev/video4:mwr \
--device /dev/video5:/dev/video5:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
ghcr.io/pinto0309/openvino2tensorflow:latest

sudo chmod 777 /dev/video4 && python3 demo_video.py
"""

import numpy as np
import cv2
import os
import argparse
from math import cos, sin
from PIL import Image
import onnxruntime
import numba as nb

idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]


def softmax(x):
    x -= np.max(x,axis=1, keepdims=True)
    a = np.exp(x)
    b = np.sum(np.exp(x), axis=1, keepdims=True)
    return a/b


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def resize_and_pad(src, size, pad_color=0):
    img = src.copy()
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = \
            np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = \
            np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3
    scaled_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp
    )
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return scaled_img


@nb.njit('i8[:](f4[:,:],f4[:], f4, b1)', fastmath=True, cache=True)
def nms_cpu(boxes, confs, nms_thresh, min_mode):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def main(args):
    yolov4_head_H = 480
    yolov4_head_W = 640
    whenet_H = 224
    whenet_W = 224

    # YOLOv4-Head
    yolov4_model_name = 'yolov4_headdetection'
    yolov4_head = onnxruntime.InferenceSession(
        f'saved_model_{whenet_H}x{whenet_W}/{yolov4_model_name}_{yolov4_head_H}x{yolov4_head_W}.onnx'
    )
    yolov4_head_input_name = yolov4_head.get_inputs()[0].name
    yolov4_head_output_names = [output.name for output in yolov4_head.get_outputs()]
    yolov4_head_output_shapes = [output.shape for output in yolov4_head.get_outputs()]
    assert yolov4_head_output_shapes[0] == [1, 18900, 1, 4] # boxes[N, num, classes, boxes]
    assert yolov4_head_output_shapes[1] == [1, 18900, 1]    # confs[N, num, classes]

    # WHENet
    whenet_input_name = None
    whenet_output_names = None
    whenet_output_shapes = None
    if args.whenet_mode == 'onnx':
        whenet = onnxruntime.InferenceSession(
            f'saved_model_{whenet_H}x{whenet_W}/model_float32.onnx'
        )
        whenet_input_name = whenet.get_inputs()[0].name
        whenet_output_names = [output.name for output in whenet.get_outputs()]
        whenet_output_shapes = [output.shape for output in whenet.get_outputs()]
        assert whenet_output_shapes[0] == [1, 120] # yaw
        assert whenet_output_shapes[1] == [1, 66]  # roll
        assert whenet_output_shapes[2] == [1, 66]  # pitch

    exec_net = None
    input_name = None
    if args.whenet_mode == 'openvino':
        from openvino.inference_engine import IECore
        model_path = f'saved_model_{whenet_H}x{whenet_W}/openvino/FP16/whenet_{whenet_H}x{whenet_W}.xml'
        ie = IECore()
        net = ie.read_network(model_path, os.path.splitext(model_path)[0] + ".bin")
        exec_net = ie.load_network(network=net, device_name='CPU', num_requests=2)
        input_name = next(iter(net.input_info))

    cap_width = int(args.height_width.split('x')[1])
    cap_height = int(args.height_width.split('x')[0])
    if args.device.isdecimal():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    WINDOWS_NAME = 'Demo'
    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS_NAME, cap_width, cap_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ============================================================= YOLOv4
        conf_thresh = 0.60
        nms_thresh = 0.50

        # Resize
        resized_frame = resize_and_pad(
            frame,
            (yolov4_head_H, yolov4_head_W)
        )
        width = resized_frame.shape[1]
        height = resized_frame.shape[0]
        # BGR to RGB
        rgb = resized_frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # normalize to [0, 1] interval
        chw = np.asarray(chw / 255., dtype=np.float32)
        # hwc --> nhwc
        nchw = chw[np.newaxis, ...]

        boxes, confs = yolov4_head.run(
            output_names = yolov4_head_output_names,
            input_feed = {yolov4_head_input_name: nchw}
        )
        # [1, boxcount, 1, 4] --> [boxcount, 4]
        boxes = boxes[0][:, 0, :]
        # [1, boxcount, 1] --> [boxcount]
        confs = confs[0][:, 0]

        argwhere = confs > conf_thresh
        boxes = boxes[argwhere, :]
        confs = confs[argwhere]
        # nms
        heads = []
        keep = nms_cpu(
            boxes=boxes,
            confs=confs,
            nms_thresh=nms_thresh,
            min_mode=False
        )
        if (keep.size > 0):
            boxes = boxes[keep, :]
            confs = confs[keep]
            for k in range(boxes.shape[0]):
                heads.append(
                    [
                        int(boxes[k, 0] * width),
                        int(boxes[k, 1] * height),
                        int(boxes[k, 2] * width),
                        int(boxes[k, 3] * height),
                        confs[k],
                    ]
                )

        canvas = resized_frame.copy()
        # ============================================================= WHENet
        croped_resized_frame = None
        if len(heads) > 0:
            for head in heads:
                x_min = head[0]
                y_min = head[1]
                x_max = head[2]
                y_max = head[3]

                # enlarge the bbox to include more background margin
                y_min = max(0, y_min - abs(y_min - y_max) / 10)
                y_max = min(resized_frame.shape[0], y_max + abs(y_min - y_max) / 10)
                x_min = max(0, x_min - abs(x_min - x_max) / 5)
                x_max = min(resized_frame.shape[1], x_max + abs(x_min - x_max) / 5)
                x_max = min(x_max, resized_frame.shape[1])
                croped_frame = resized_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                # h,w -> 224,224
                croped_resized_frame = cv2.resize(croped_frame, (whenet_W, whenet_H))
                # bgr --> rgb
                rgb = croped_resized_frame[..., ::-1]
                # Normalization
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                rgb = ((rgb / 255.0) - mean) / std
                # hwc --> chw
                chw = rgb.transpose(2, 0, 1)
                # chw --> nchw
                nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)

                yaw = 0.0
                pitch = 0.0
                roll = 0.0
                if args.whenet_mode == 'onnx':
                    yaw, roll, pitch = whenet.run(
                        output_names = whenet_output_names,
                        input_feed = {whenet_input_name: nchw}
                    )
                elif args.whenet_mode == 'openvino':
                    output = exec_net.infer(inputs={input_name: nchw})
                    yaw = output['yaw_new/BiasAdd/Add']
                    pitch = output['pitch_new/BiasAdd/Add']
                    roll = output['roll_new/BiasAdd/Add']

                yaw = np.sum(softmax(yaw) * idx_tensor_yaw, axis=1) * 3 - 180
                pitch = np.sum(softmax(pitch) * idx_tensor, axis=1) * 3 - 99
                roll = np.sum(softmax(roll) * idx_tensor, axis=1) * 3 - 99
                yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

                print(f'yaw: {yaw}, pitch: {pitch}, roll: {roll}')

                # BBox draw
                deg_norm = 1.0 - abs(yaw / 180)
                blue = int(255 * deg_norm)
                cv2.rectangle(
                    canvas,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    color=(blue, 0, 255-blue),
                    thickness=2
                )

                # Draw
                draw_axis(
                    canvas,
                    yaw,
                    pitch,
                    roll,
                    tdx=(x_min+x_max)/2,
                    tdy=(y_min+y_max)/2,
                    size=abs(x_max-x_min)//2
                )
                cv2.putText(canvas, f'yaw: {np.round(yaw)}', (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
                cv2.putText(canvas, f'pitch: {np.round(pitch)}', (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
                cv2.putText(canvas, f'roll: {np.round(roll)}', (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)

            # cv2.imshow('Face', croped_resized_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        cv2.imshow(WINDOWS_NAME, canvas)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--whenet_mode",
        type=str,
        default='onnx',
        choices=['onnx', 'openvino'],
        help='Choose whether to infer WHENet with ONNX or OpenVINO. Default: onnx',
    )
    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        "--height_width",
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    args = parser.parse_args()
    main(args)