#!/usr/bin/env python3

import argparse
import time
import cv2
import copy
import numpy as np
import onnxruntime
from typing import Tuple, Optional, List

WINDOW_NAME = 'test'


def draw_caption(
    image: np.ndarray,
    box: Tuple,
    caption: str,
) -> None:

    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def resize_and_pad(
    src: np.ndarray,
    size: Tuple,
    pad_color: Optional[int]=0,
) -> np.ndarray:

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
        h_ratio = sh/h
        w_ratio = sw/w
        resize_ratio = min(h_ratio, w_ratio)
        new_h = np.round(h*resize_ratio).astype(int)
        new_w = np.round(w*resize_ratio).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
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


def nms_cpu(
    boxes: np.ndarray,
    confs: np.ndarray,
    nms_thresh: float,
    min_mode: bool,
) -> np.ndarray:

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_detection_model",
        default='yolov4_headdetection_480x640.onnx',
    )
    parser.add_argument(
        "--face_landmark_model",
        default='face_landmark_Nx3x160x160.onnx',
    )
    parser.add_argument(
        "--output",
        help="File path of output movie.",
        type=str,
        default='output.mp4',
    )
    args = parser.parse_args()

    face_detection_model = args.face_detection_model
    face_landmark_model = args.face_landmark_model

    # Initialize window
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Video capture
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)


    # Load model
    ### Face Detection - YOLOv4
    session_option_det = onnxruntime.SessionOptions()
    session_option_det.log_severity_level = 3
    face_detection_sess = onnxruntime.InferenceSession(
        face_detection_model,
        sess_options=session_option_det,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    ### Face Landmark - ShuffleNetV2 0.75
    session_option_land = onnxruntime.SessionOptions()
    session_option_land.log_severity_level = 3
    face_landmark_sess = onnxruntime.InferenceSession(
        face_landmark_model,
        sess_options=session_option_land,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    # Face Detection - Input information
    face_detection_inputs = face_detection_sess.get_inputs()
    dn, dc, dh, dw = face_detection_inputs[0].shape

    # Face Landmark - Input information
    face_landmark_inputs = face_landmark_sess.get_inputs()
    ln, lc, lh, lw = face_landmark_inputs[0].shape

    # MP4 writer
    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        start = time.perf_counter()
        image = copy.deepcopy(frame)

        # Inference - Face Detection =================================== YOLOv4
        conf_thresh = 0.60
        nms_thresh = 0.50

        # Resize
        resized_frame = resize_and_pad(image, (dh, dw))
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
        # inference
        boxes, confs = face_detection_sess.run(
            None,
            {face_detection_inputs[0].name: nchw},
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
                """
                heads = [
                    [x1,y1,x2,y2,score],
                    [x1,y1,x2,y2,score],
                    [x1,y1,x2,y2,score],
                    ...
                ]
                """
                heads.append(
                    [
                        int(boxes[k, 0] * width),
                        int(boxes[k, 1] * height),
                        int(boxes[k, 2] * width),
                        int(boxes[k, 3] * height),
                        confs[k],
                    ]
                )

        if len(heads) > 0:
            head_images = None
            head_sizes = []
            for head in heads:
                x_min = head[0]
                y_min = head[1]
                x_max = head[2]
                y_max = head[3]

                # Face Bounding Box drawing
                cv2.rectangle(
                    frame,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    color=(255, 0, 0),
                    thickness=2
                )

                # Pre-Process - Face Landmark ========================= ShuffleNetV2 0.75
                head_width = x_max - x_min
                head_height = y_max - y_min
                croped_frame = resized_frame[y_min:y_max,x_min:x_max,:]
                image = cv2.resize(croped_frame, (lw, lh))
                rgb = image[..., ::-1]
                chw = rgb.transpose(2, 0, 1)
                chw = chw.astype(np.float32)
                nchw = chw[np.newaxis, ...]

                # Generation of image batches for landmark detection
                if head_images is None:
                    head_images = nchw
                else:
                    head_images = np.concatenate(
                        arrays=[head_images, nchw],
                        axis=0,
                    )
                head_sizes.append([head_width, head_height])

            # Inference - Face Landmark
            """
            outputs[0].shape
            (1, 4)
            outputs[1].shape
            (1, 3)
            outputs[2].shape
            (1, 136)
            """
            landmark_outputs = face_landmark_sess.run(
                None,
                {face_landmark_inputs[0].name: head_images},
            )
            inference_time = (time.perf_counter() - start) * 1000

            # Draw Face alignment
            """
            landmarks = [N, 68, 2] ... N x 68keypoints x XY
            """
            landmarks = np.array(landmark_outputs[2]).reshape([len(heads), -1, 2])
            for landmark, head_size in zip(landmarks, head_sizes):
                for keypoints in landmark:
                    center = (int(keypoints[0] * head_size[0]), int(keypoints[1] * head_size[1]))
                    cv2.circle(croped_frame, center, 7, (246, 250, 250), -1)
                    cv2.circle(croped_frame, center, 2, (255, 209, 0), 2)
            frame[y_min:y_max,x_min:x_max] = croped_frame

        # Calc fps
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = f'{fps_text} {avg_text}'
        draw_caption(frame, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
