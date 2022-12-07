#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-mo', '--movie', type=str, default=None)
    parser.add_argument('-wt', '--width', help='cap width', type=int, default=960)
    parser.add_argument('-ht', '--height', help='cap height', type=int, default=540)

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='saved_model/e2epose_resnet101_1x3x512x512.onnx',
    )
    parser.add_argument(
        '-s',
        '--score_th',
        type=float,
        default=0.4,
        help='Class confidence',
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        '-c',
        '--use_cuda',
        action='store_true',
    )
    device_group.add_argument(
        '-trt',
        '--use_tensorrt',
        action='store_true',
    )

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_name, input_size,image, score_th=0.5):
    # ONNX Infomation
    input_width = input_size[3]
    input_height = input_size[2]
    image_height, image_width = image.shape[0], image.shape[1]

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    results = onnx_session.run(None, {input_name: input_image})

    # Post process
    kpt, pv = results
    pv = np.reshape(pv[0], [-1])
    kpt = kpt[0][pv >= score_th]
    kpt[:, :, -1] *= image_height
    kpt[:, :, -2] *= image_width
    kpt[:, :, -3] *= 2
    ret = []
    for human in kpt:
        mask = np.stack(
            [(human[:, 0] >= score_th).astype(np.float32)],
            axis=-1,
        )
        human *= mask
        human = np.stack([human[:, _ii] for _ii in [1, 2, 0]], axis=-1)
        ret.append({
            'keypoints': np.reshape(human, [-1]).tolist(),
            'category_id': 1
        })

    return ret


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    score_th = args.score_th
    use_cuda = args.use_cuda
    use_tensorrt = args.use_tensorrt

    # Initialize video capture
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # Load model
    providers = None
    if not use_cuda and not use_tensorrt:
        providers = [
            'CPUExecutionProvider',
        ]
    elif use_cuda:
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif use_tensorrt:
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
    )
    input_name = onnx_session.get_inputs()[0].name
    input_size = onnx_session.get_inputs()[0].shape

    while True:
        start_time = time.time()

        # Capture
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        results = run_inference(
            onnx_session,
            input_name,
            input_size,
            frame,
            score_th=score_th,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            results,
        )

        cv2.imshow('E2Pose ONNX Sample', debug_image)
        video_writer.write(debug_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def draw_debug(image, elapsed_time, results):
    debug_image = copy.deepcopy(image)

    # 00:鼻(nose), 01:左目(left eye), 02:右目(right eye), 03:左耳(left ear)
    # 04:右耳(right ear), 05:左肩(left shoulder), 06:右肩(right shoulder)
    # 07:左肘(left elbow), 08:右肘(right elbow), 09:左手首(left wrist)
    # 10:右手首(right wrist), 11:左腰(left waist), 12:右腰(right waist),
    # 13:左膝(left knee), 14:右膝(right knee), 15:左足首(left ankle)
    # 16:右足首(right ankle)

    connect_list = [
        [0, 1],  # 00:鼻(nose) -> 01:左目(left eye)
        [0, 2],  # 00:鼻(nose) -> 02:右目(right eye)
        [1, 3],  # 01:左目(left eye) -> 03:左耳(left ear)
        [2, 4],  # 02:右目(right eye) -> 04:右耳(right ear)
        [3, 5],  # 03:左耳(left ear) -> 05:左肩(left shoulder)
        [4, 6],  # 04:右耳(right ear) -> 06:右肩(right shoulder)
        [5, 6],  # 05:左肩(left shoulder) -> 06:右肩(right shoulder)
        [5, 7],  # 05:左肩(left shoulder)  -> 07:左肘(left elbow)
        [7, 9],  # 07:左肘(left elbow) -> 09:左手首(left wrist)
        [6, 8],  # 06:右肩(right shoulder) -> 08:右肘(right elbow)
        [8, 10],  # 08:右肘(right elbow) -> 10:右手首(right wrist)
        [5, 11],  # 05:左肩(left shoulder) -> 11:左腰(left waist)
        [6, 12],  # 06:右肩(right shoulder) -> 12:右腰(right waist)
        [11, 12],  # 11:左腰(left waist) -> 12:右腰(right waist)
        [11, 13],  # 11:左腰(left waist) -> 13:左膝(left knee)
        [13, 15],  # 13:左膝(left knee) -> 15:左足首(left ankle)
        [12, 14],  # 12:右腰(right waist) -> 14:右膝(right knee),
        [14, 16],  # 14:右膝(right knee) -> 16:右足首(right ankle)
    ]

    for result in results:
        keypoints_iter = iter(result['keypoints'])

        keypoint_list = []
        for cx, cy, _ in zip(keypoints_iter, keypoints_iter, keypoints_iter):
            cx = int(cx)
            cy = int(cy)
            keypoint_list.append([cx, cy])

        for keypoint in keypoint_list:
            cx = keypoint[0]
            cy = keypoint[1]
            if cx > 0 and cy > 0:
                cv2.circle(
                    debug_image,
                    (cx, cy),
                    3,
                    (0, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )

        for connect in connect_list:
            cx1 = keypoint_list[connect[0]][0]
            cy1 = keypoint_list[connect[0]][1]
            cx2 = keypoint_list[connect[1]][0]
            cy2 = keypoint_list[connect[1]][1]
            if cx1 > 0 and cy1 > 0 and cx2 > 0 and cy2 > 0:
                cv2.line(
                    debug_image,
                    (cx1, cy1),
                    (cx2, cy2),
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

    # Elapsed time
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
