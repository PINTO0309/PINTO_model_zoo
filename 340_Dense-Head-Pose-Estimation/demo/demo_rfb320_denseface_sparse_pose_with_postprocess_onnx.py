#!/usr/bin/env python

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
from typing import Tuple, List


def run_inference_rfb(
    onnx_session: onnxruntime.InferenceSession,
    input_name: str,
    input_size: List[int],
    image: np.ndarray,
    score_th: float=0.8,
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    # ONNX Infomation
    input_width = input_size[3]
    input_height = input_size[2]
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = input_image[..., ::-1]
    input_image = input_image.transpose(2, 0, 1)[np.newaxis, ...]
    input_image = (input_image - 127.5) / 127.5
    input_image = input_image.astype('float32')
    # Inference
    results = onnx_session.run(
        None,
        {input_name: input_image},
    )[0]
    # Post process
    bboxes, scores, class_ids = [], [], []
    for batchno_classid_score_x1y1x2y2 in results:
        bbox = batchno_classid_score_x1y1x2y2[-4:].tolist()
        class_id = int(batchno_classid_score_x1y1x2y2[1])
        score = batchno_classid_score_x1y1x2y2[2]
        if score_th > score:
            continue
        image_height, image_width = image.shape[0], image.shape[1]
        bbox[0] = int(bbox[0] * image_width)
        bbox[1] = int(bbox[1] * image_height)
        bbox[2] = int(bbox[2] * image_width)
        bbox[3] = int(bbox[3] * image_height)
        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)
    return np.asarray(bboxes), np.asarray(scores), np.asarray(class_ids)


def run_inference_face_alignment(
    onnx_session: onnxruntime.InferenceSession,
    input_name: str,
    input_size: List[int],
    frame: np.ndarray,
    faces: np.ndarray,
    mode='pose',
) -> Tuple[List[np.ndarray],List[np.ndarray]]:
    image = copy.deepcopy(frame)
    inputs = []
    Ms = []
    for face in faces:
        input_shape_tp = tuple(input_size[2:])
        edge_size = input_shape_tp[-1]
        trans_distance = edge_size / 2.0
        maximum_edge = max(face[2:4] - face[:2]) * 2.7
        scale = edge_size * 2.0 / maximum_edge
        center = (face[2:4] + face[:2]) / 2.0
        cx, cy = trans_distance - scale * center
        M = np.array([[scale, 0, cx], [0, scale, cy]])
        cropped = cv.warpAffine(image, M, input_shape_tp, borderValue=0.0)
        rgb = cropped[:, :, ::-1].astype(np.float32)
        cv.normalize(rgb, rgb, alpha=-1, beta=1, norm_type=cv.NORM_MINMAX)
        inp = rgb.transpose(2, 0, 1)
        inputs.append(inp)
        Ms.append(M)
    # Inference
    camera_matrixes = []
    landmarks = []
    if len(inputs) > 0:
        camera_matrixes, landmarks = onnx_session.run(
            None,
            {input_name: np.asarray(inputs, dtype=np.float32)},
        )
    faces_points = []
    Rs = []
    for camera_matrix, landmark, M_ in zip(camera_matrixes, landmarks, Ms):
        iM = cv.invertAffineTransform(M_)
        R = copy.deepcopy(camera_matrix)
        points = copy.deepcopy(landmark)
        if mode in ['sparse', 'pose']:
            points *= iM[0, 0]
            points += iM[:, -1]
        elif mode in ['dense']:
            points *= iM[0][0]
            points[:, :2] += iM[:, -1]
        faces_points.append(points)
        Rs.append(R)
    return faces_points, Rs


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([x, y, z])


def build_projection_matrix(rear_size, factor=np.sqrt(2)):
    rear_depth = 0
    front_size = front_depth = factor * rear_size
    projections = np.array([
        [-rear_size, -rear_size, rear_depth],
        [-rear_size, rear_size, rear_depth],
        [rear_size, rear_size, rear_depth],
        [rear_size, -rear_size, rear_depth],
        [-front_size, -front_size, front_depth],
        [-front_size, front_size, front_depth],
        [front_size, front_size, front_depth],
        [front_size, -front_size, front_depth],
    ], dtype=np.float32)
    return projections


def draw_projection(frame, R, landmarks, color, thickness=2):
    # build projection matrix
    radius = np.max(np.max(landmarks, 0) - np.min(landmarks, 0)) // 2
    projections = build_projection_matrix(radius)
    # refine rotate matrix
    rotate_matrix = R[:, :2]
    rotate_matrix[:, 1] *= -1
    # 3D -> 2D
    center = np.mean(landmarks[:27], axis=0)
    points = projections @ rotate_matrix + center
    points = points.astype(np.int32)
    # draw poly
    cv.polylines(frame, np.take(points, [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [4, 5], [5, 6], [6, 7], [7, 4]
    ], axis=0), False, color, thickness, cv.LINE_AA)


def draw_poly(frame, landmarks, color=(128, 255, 255), thickness=1):
    cv.polylines(frame, [
        landmarks[:17],
        landmarks[17:22],
        landmarks[22:27],
        landmarks[27:31],
        landmarks[31:36]
    ], False, color, thickness=thickness)
    cv.polylines(frame, [
        landmarks[36:42],
        landmarks[42:48],
        landmarks[48:60],
        landmarks[60:]
    ], True, color, thickness=thickness)


def sparse(frame, landmarks, params, color):
    landmarks = np.round(landmarks).astype(np.int)
    _ = [
        cv.circle(frame, tuple(p), 2, color, 0, cv.LINE_AA) for p in landmarks
    ]
    draw_poly(frame, landmarks, color=color)


def dense(frame, landmarks, params, color):
    landmarks = np.round(landmarks).astype(np.int)
    _ = [
        cv.circle(frame, tuple(p), 1, color, 0, cv.LINE_AA) for p in landmarks[::6, :2]
    ]


def pose(frame, landmarks, params, color):
    # rotate matrix
    R = params[:3, :3].copy()
    # decompose matrix to ruler angle
    # euler = rotationMatrixToEulerAngles(R)
    # print(f"Pitch: {euler[0]}; Yaw: {euler[1]}; Roll: {euler[2]};")
    draw_projection(frame, R, landmarks, color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-mov',
        '--movie',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-modrfb',
        '--model_rfb',
        type=str,
        default='RFB-320_240x320_post.onnx',
    )
    parser.add_argument(
        '-modden',
        '--model_dense',
        type=str,
        default='dense_face_Nx3x120x120.onnx',
    )
    parser.add_argument(
        '-modspa',
        '--model_sparse',
        type=str,
        default='sparse_face_Nx3x120x120.onnx',
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='pose',
        choices=['pose','sparse','dense'],
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cpu',
        choices=['cpu','cuda','tensorrt'],
    )
    parser.add_argument(
        '-s',
        '--score_th',
        type=float,
        default=0.7,
    )
    args = parser.parse_args()
    device: int = args.device
    movie: str = args.movie
    model_rfb: str = args.model_rfb
    model_dense: str = args.model_dense
    model_sparse: str = args.model_sparse
    mode: str = args.mode
    provider: str = args.provider
    score_th: float = args.score_th

    # Initialize video capture
    cap_device = device
    if movie is not None:
        cap_device = movie
    cap = cv.VideoCapture(cap_device)
    cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # Load model
    providers = []
    if provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif provider == 'tensorrt':
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

    onnx_session_rfb = onnxruntime.InferenceSession(
        path_or_bytes=model_rfb,
        providers=providers,
    )
    input_name_rfb: str = onnx_session_rfb.get_inputs()[0].name
    input_size_rfb: List[int] = onnx_session_rfb.get_inputs()[0].shape

    onnx_session_align = None
    if mode in ['pose','sparse']:
        onnx_session_align = onnxruntime.InferenceSession(
            path_or_bytes=model_sparse,
            providers=providers,
        )
    elif mode in ['dense']:
        onnx_session_align = onnxruntime.InferenceSession(
            path_or_bytes=model_dense,
            providers=providers,
        )
    input_name_align: str = onnx_session_align.get_inputs()[0].name
    input_size_align: List[int] = onnx_session_align.get_inputs()[0].shape

    alignment_draw_func = None
    if mode == 'pose':
        alignment_draw_func = pose
    elif mode == 'sparse':
        alignment_draw_func = sparse
    elif mode == 'dense':
        alignment_draw_func = dense

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # FaceDetection inference execution
        faces, scores, class_ids = run_inference_rfb(
            onnx_session_rfb,
            input_name_rfb,
            input_size_rfb,
            frame,
            score_th=score_th,
        )

        # FaceAlignment inference execution
        landmarks, Rs = run_inference_face_alignment(
            onnx_session_align,
            input_name_align,
            input_size_align,
            frame,
            faces,
            mode=mode,
        )

        # FaceAlignment drawing
        for landmark, R in zip(landmarks, Rs):
            alignment_draw_func(
                frame=debug_image,
                landmarks=landmark,
                params=R,
                color=(224, 255, 255),
            )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            f'Elapsed Time: {elapsed_time * 1000:.1f} ms',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        video_writer.write(debug_image)
        cv.imshow(f'Dense-Head-Pose-Estimation ({mode}) ONNX', debug_image)
        key = cv.waitKey(1) if movie is None or movie[-4:] == '.mp4' else cv.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()