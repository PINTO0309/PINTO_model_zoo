#!/usr/bin/env python

from __future__ import annotations
import os
import copy
import cv2
import csv
import datetime
import time
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from collections import deque
from typing import Tuple, Optional, List, Deque
from math import cos, sin
from scipy.spatial import distance
from dataclasses import dataclass
from bbalg import state_verdict

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    is_hand_used: bool = False

@dataclass(frozen=False)
class TrackedBox:
    box: Box
    id: int
    lost: int = 0  # 追跡を見失ったフレーム数
    looking_duration: int = 90 # 30 frame * 3 sec
    looking_duration_long: int = looking_duration * 2 # 30 frame * 6 sec
    looking_duration_short: int = looking_duration # 30 frame * 3 sec
    looking_history_long = deque(maxlen=looking_duration_long)
    looking_history_short = deque(maxlen=looking_duration_short)

class HeadTracker:
    def __init__(self, max_distance=50, max_lost=30, looking_duration=90):
        self.tracked_heads: dict[int, TrackedBox] = {}  # 辞書に変更
        self.next_id = 0
        self.max_distance = max_distance  # 中心点同士の距離の閾値
        self.max_lost = max_lost  # オブジェクトを見失っても保持するフレーム数
        self.looking_duration = looking_duration  # 注視判定時間

    def update_trackers(self, head_boxes: List[Box]):
        # 現在追跡中のオブジェクトの中心点を取得
        tracked_centroids = [(t.id, (t.box.cx, t.box.cy)) for t in self.tracked_heads.values()]
        # 新しく検出されたバウンディングボックスの中心点を取得
        new_centroids = [(box.cx, box.cy) for box in head_boxes]

        # マッチング処理
        matched, unmatched_tracked, unmatched_new = self.match_objects(tracked_centroids, new_centroids)

        # 対応したトラッカーを更新
        for tracked_idx, new_idx in matched:
            tracked_id = tracked_centroids[tracked_idx][0]
            self.tracked_heads[tracked_id].box = head_boxes[new_idx]
            self.tracked_heads[tracked_id].lost = 0  # 見失いカウンタをリセット

        # 対応しなかったトラッカーをカウントアップ、見失いが多いものを削除
        for tracked_idx in unmatched_tracked:
            tracked_id = tracked_centroids[tracked_idx][0]
            self.tracked_heads[tracked_id].lost += 1
            if self.tracked_heads[tracked_id].lost > self.max_lost:
                del self.tracked_heads[tracked_id]  # 追跡対象から削除

        # 新しい検出結果を追加
        for new_idx in unmatched_new:
            self.tracked_heads[self.next_id] = TrackedBox(
                box=head_boxes[new_idx],
                id=self.next_id,
                looking_duration=self.looking_duration
            )
            self.next_id += 1

        return list(self.tracked_heads.values())

    def match_objects(self, tracked_centroids, new_centroids):
        matched = []
        unmatched_tracked = list(range(len(tracked_centroids)))
        unmatched_new = list(range(len(new_centroids)))

        if len(tracked_centroids) > 0 and len(new_centroids) > 0:
            distance_matrix = distance.cdist(
                [tc[1] for tc in tracked_centroids],
                new_centroids,
                metric='euclidean'
            )

            for tracked_idx in range(len(tracked_centroids)):
                min_distance_idx = np.argmin(distance_matrix[tracked_idx])
                min_distance = distance_matrix[tracked_idx][min_distance_idx]

                if min_distance < self.max_distance:
                    matched.append((tracked_idx, min_distance_idx))

                    # `unmatched_tracked` から削除
                    if tracked_idx in unmatched_tracked:
                        unmatched_tracked.remove(tracked_idx)

                    # `unmatched_new` から削除
                    if min_distance_idx in unmatched_new:
                        unmatched_new.remove(min_distance_idx)

        return matched, unmatched_tracked, unmatched_new

    def stack_looking_history(self, tracked_id: int, state: bool):
        if tracked_id in self.tracked_heads:
            self.tracked_heads[tracked_id].looking_history_long.append(state)
            self.tracked_heads[tracked_id].looking_history_short.append(state)

    def get_state_start(self, tracked_id: int) -> bool:
        if tracked_id in self.tracked_heads:
            state_interval, state_start, state_end = state_verdict(
                long_tracking_history=self.tracked_heads[tracked_id].looking_history_long,
                short_tracking_history=self.tracked_heads[tracked_id].looking_history_short,
            )
            return state_start
        return False

class GoldYOLOONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = [
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
    ):
        """GoldYOLOONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for GoldYOLO

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
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
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOv7ONNX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )[0]

        # PostProcess
        result_boxes = \
            self.__postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes


    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        result_scores: np.ndarray
            Predicted box confs: [N, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_y1x1y2x2_score: float32[N,7]
        """
        result_boxes: List[Box] = []
        if len(boxes) > 0:
            scores = boxes[:, 6:7]
            keep_idxs = scores[:, 0] > self.class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]
            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(box[2], 0) * image_width / self.input_shapes[0][3])
                    y_min = int(max(box[3], 0) * image_height / self.input_shapes[0][2])
                    x_max = int(min(box[4], self.input_shapes[0][3]) * image_width / self.input_shapes[0][3])
                    y_max = int(min(box[5], self.input_shapes[0][2]) * image_height / self.input_shapes[0][2])
                    result_boxes.append(
                        Box(
                            classid=0,
                            score=score,
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=(x_min + x_max) // 2,
                            cy=(y_min + y_max) // 2,
                        )
                    )
        return result_boxes

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
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
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)
    return img

def is_looking_at_camera_with_angles(
    box: Box,
    yaw: float,
    pitch: float,
    image_width=640,
    image_height=480,
    yaw_threshold=15,
    pitch_threshold=15,
):
    """
    バウンディングボックスの位置とYaw, Pitchからカメラ中心を見ているかを角度補正して判定する関数

    Parameters:
    - box: Boxオブジェクト（cx, cyを含む）
    - yaw: Yaw（ヨー）の角度（度数）
    - pitch: Pitch（ピッチ）の角度（度数）
    - image_width: 画像の横幅（デフォルト640）
    - image_height: 画像の縦幅（デフォルト480）
    - yaw_threshold: Yawの閾値（この範囲内ならカメラの中心を見ているとみなす）
    - pitch_threshold: Pitchの閾値（この範囲内ならカメラの中心を見ているとみなす）

    Returns:
    - True: カメラの中心を見ていると判断
    - False: カメラの中心を見ていない
    """

    # カメラの中心座標
    camera_center_x = image_width // 2
    camera_center_y = image_height // 2

    # バウンディングボックスの中心座標
    box_center_x = box.cx
    box_center_y = box.cy

    # 画面の幅と高さの比率を使用して、バウンディングボックスの中心がどれだけ左右にズレているか計算
    horizontal_offset_ratio = (box_center_x - camera_center_x) / (image_width / 2)  # -1.0から1.0の範囲に収める
    vertical_offset_ratio = (box_center_y - camera_center_y) / (image_height / 2)  # -1.0から1.0の範囲に収める

    # Yawの補正を適用（顔が画面の左にあるなら右向きに、右にあるなら左向きに補正）
    corrected_yaw = yaw - horizontal_offset_ratio * 30  # 最大±30度の補正を適用

    # Pitchの補正を適用（顔が上にあるなら下向きに、下にあるなら上向きに補正）
    corrected_pitch = pitch + vertical_offset_ratio * 20  # 最大±20度の補正を適用

    # 補正後のYawとPitchが閾値内かどうかで判断
    is_yaw_in_range = abs(corrected_yaw) <= yaw_threshold
    is_pitch_in_range = abs(corrected_pitch) <= pitch_threshold

    # 両方の条件を満たせばカメラの中心を見ていると判断
    if is_yaw_in_range and is_pitch_in_range:
        return True
    else:
        return False

class LogWriter:
    def __init__(self, base_filename, max_lines=18000, header_row=None):
        self.base_filename = base_filename
        self.max_lines = max_lines
        self.current_line_count = 0  # ログ行のカウント（ヘッダを除外）
        self.current_file_index = 1
        self.log_file = None
        self.csv_writer = None
        self.header_row = header_row  # ヘッダ行
        self.start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # 初回のログファイルを開く
        self._open_new_file()

    def _open_new_file(self):
        if self.log_file:
            self.log_file.close()

        # タイムスタンプと連番を含む新しいファイル名を生成
        filename = f"{self.start_time}_{self.current_file_index:03d}_{self.base_filename}"
        self.log_file = open(filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.current_line_count = 0  # 新しいファイルではログ行を0からスタート
        self.current_file_index += 1

        # 新しいファイルごとにヘッダ行を出力
        if self.header_row:
            self.csv_writer.writerow(self.header_row)

    def write_row(self, row):
        """ログ行を書き込み、行カウントを更新"""
        if self.current_line_count >= self.max_lines:
            # 最大行数に達したら新しいファイルを開き、ヘッダを書き込む
            self._open_new_file()

        # ログ行を書き込み、行カウントを増やす
        self.csv_writer.writerow(row)
        self.current_line_count += 1

    def close(self):
        """ファイルを閉じる"""
        if self.log_file:
            self.log_file.close()

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gold_yolo_l_head_post_0277_0.5353_1x3x480x640.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    parser.add_argument(
        '-l',
        '--enable_log',
        action='store_true',
        help='Enable CSV logging of detection results.',
    )
    parser.add_argument(
        '-mi',
        '--max_logging_instances',
        type=int,
        default=20,
        help='Max logging instances. Default: 20',
    )
    parser.add_argument(
        '-mr',
        '--max_logging_rows',
        type=int,
        default=18000,
        help='Max logging rows. Default: 18000 (10 min)',
    )
    parser.add_argument(
        '-ld',
        '--looking_duration',
        type=int,
        default=3,
        help='Looking duration. Default: 3 (3 sec)',
    )
    args = parser.parse_args()

    model = GoldYOLOONNX(
        model_path=args.model,
        class_score_th=0.65,
    )

    session_option = onnxruntime.SessionOptions()
    session_option.log_severity_level = 3
    onnx_session = onnxruntime.InferenceSession(
        'sixdrepnet360_Nx3x224x224_full.onnx',
        sess_options=session_option,
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
        ]
    )

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    cap = cv2.VideoCapture(
        int(args.video) if is_parsable_to_int(args.video) else args.video
    )
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(image_width, image_height),
    )

    enable_log: bool = args.enable_log
    max_logging_instances: int = args.max_logging_instances
    max_logging_rows: int = args.max_logging_rows
    looking_duration: int = args.looking_duration

    head_tracker = HeadTracker(max_distance=50, max_lost=30, looking_duration=int(looking_duration * cap_fps))

    log_writer: LogWriter = None
    if enable_log:
        header_row = ['timestamp']
        for idx in range(max_logging_instances):
            header_row += [f'head_{idx+1}', f'looking_{idx+1}']
        log_writer = LogWriter(base_filename="log.csv", max_lines=max_logging_rows, header_row=header_row)

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)

        start_time = time.time()
        boxes = model(debug_image)

        if len(boxes) > 0:
            image_height = debug_image.shape[0]
            image_width = debug_image.shape[1]

            x1y1x2y2cxcys: List = []
            normalized_image_rgbs: List = []
            for box in boxes:
                w: int = abs(box.x2 - box.x1)
                h: int = abs(box.y2 - box.y1)
                ew: float = w * 1.2
                eh: float = h * 1.2
                ex1 = int(box.cx - ew / 2)
                ex2 = int(box.cx + ew / 2)
                ey1 = int(box.cy - eh / 2)
                ey2 = int(box.cy + eh / 2)

                ex1 = ex1 if ex1 >= 0 else 0
                ex2 = ex2 if ex2 <= image_width else image_width
                ey1 = ey1 if ey1 >= 0 else 0
                ey2 = ey2 if ey2 <= image_height else image_height

                inference_image = copy.deepcopy(debug_image)
                head_image_bgr = inference_image[ey1:ey2, ex1:ex2, :]
                resized_image_bgr = cv2.resize(head_image_bgr, (256, 256))
                cropped_image_bgr = resized_image_bgr[16:240, 16:240, :]

                # inference
                cropped_image_rgb: np.ndarray = cropped_image_bgr[..., ::-1]
                normalized_image_rgb: np.ndarray = (cropped_image_rgb / 255.0 - mean) / std
                normalized_image_rgb = normalized_image_rgb.transpose(2,0,1)
                normalized_image_rgb: np.ndarray = normalized_image_rgb.astype(np.float32)

                x1y1x2y2cxcys.append([box.x1, box.y1, box.x2, box.y2, box.cx, box.cy])
                normalized_image_rgbs.append(normalized_image_rgb)

            yaw_pitch_rolls: np.ndarray = \
                onnx_session.run(
                    None,
                    {'input': np.asarray(normalized_image_rgbs, dtype=np.float32)},
                )[0]

            tracked_heads = head_tracker.update_trackers(boxes)

            elapsed_time = time.time() - start_time

            for yaw_pitch_roll, tracked_head in zip(yaw_pitch_rolls, tracked_heads):
                yaw_deg = yaw_pitch_roll[0]
                pitch_deg = yaw_pitch_roll[1]
                roll_deg = yaw_pitch_roll[2]
                x1 = tracked_head.box.x1
                y1 = tracked_head.box.y1
                x2 = tracked_head.box.x2
                y2 = tracked_head.box.y2
                cx = tracked_head.box.cx
                cy = tracked_head.box.cy

                cv2.rectangle(
                    debug_image,
                    (x1, y1),
                    (x2, y2),
                    (255,255,255),
                    2,
                )
                cv2.rectangle(
                    debug_image,
                    (x1, y1),
                    (x2, y2),
                    (0,0,255),
                    1,
                )

                looking_camera_txt = ''
                is_looking = 1 if is_looking_at_camera_with_angles(tracked_head.box, yaw_deg, pitch_deg, image_width, image_height) else 0
                head_tracker.stack_looking_history(tracked_head.id, True if is_looking == 1 else False)
                is_looking_start = head_tracker.get_state_start(tracked_head.id)

                if is_looking_start:
                    looking_camera_txt = 'Looking'
                else:
                    looking_camera_txt = ''

                if enable_log:
                    log_row = []
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y%m%d%H%M%S") + f'{now.microsecond // 1000:03d}'
                    log_row.append(timestamp)
                    log_row.append(f'{tracked_head.id}')
                    log_row.append(f'{is_looking_start}')
                    while len(log_row) < (max_logging_instances * 2 + 1):
                        log_row.append('')
                    log_writer.write_row(log_row)

                cv2.putText(
                    debug_image,
                    f'{tracked_head.id:06} {looking_camera_txt}',
                    (
                        x1,
                        y1-10 if y1-10 > 0 else 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{tracked_head.id:06} {looking_camera_txt}',
                    (
                        x1,
                        y1-10 if y1-10 > 0 else 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                draw_axis(debug_image, yaw_deg, pitch_deg, roll_deg, tdx=float(cx), tdy=float(cy), size=abs(x2-x1)//2)
        else:
            elapsed_time = time.time() - start_time
            if enable_log:
                log_row = []
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d%H%M%S") + f'{now.microsecond // 1000:03d}'
                log_row.append(timestamp)
                while len(log_row) < (max_logging_instances * 2 + 1):
                    log_row.append('')
                log_writer.write_row(log_row)

        # fps = 1 / elapsed_time
        # cv2.putText(
        #     debug_image,
        #     f'{fps:.1f} FPS (inferece + post-process)',
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )
        # cv2.putText(
        #     debug_image,
        #     f'{fps:.1f} FPS (inferece + post-process)',
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 0, 255),
        #     1,
        #     cv2.LINE_AA,
        # )

        cv2.imshow("test", debug_image)
        video_writer.write(debug_image)
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

    if enable_log:
        log_writer.close()

    if video_writer:
        video_writer.release()

    if cap:
        cap.release()

if __name__ == "__main__":
    main()
