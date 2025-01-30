#!/usr/bin/env python

from __future__ import annotations
import os
import copy
import cv2
import csv
import sys
import datetime
import time
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from collections import deque
from enum import Enum
from pprint import pprint
from typing import Tuple, Optional, List, Deque, Dict
from math import cos, sin
from scipy.spatial import distance
from dataclasses import dataclass
from bbalg import state_verdict
import importlib.util
from abc import ABC, abstractmethod
from datetime import datetime

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

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
    generation: int = -1 # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1 # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1 # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
    is_used: bool = False

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

    headtilt_duration: int = 30 # 30 frame * 1 sec
    headtilt_duration_long: int = headtilt_duration * 1 # 30 frame * 1 sec
    headtilt_duration_short: int = headtilt_duration // 2 # 30 frame * 0.5 sec
    headtilt_history_long = deque(maxlen=headtilt_duration_long)
    headtilt_history_short = deque(maxlen=headtilt_duration_short)

class HeadTracker:
    def __init__(self, max_distance=50, max_lost=30, looking_duration=90, headtilt_duration=30):
        self.tracked_heads: dict[int, TrackedBox] = {}  # 辞書に変更
        self.next_id = 0
        self.max_distance = max_distance  # 中心点同士の距離の閾値
        self.max_lost = max_lost  # オブジェクトを見失っても保持するフレーム数
        self.looking_duration = looking_duration  # 注視判定時間
        self.headtilt_duration = headtilt_duration  # 首かしげ判定時間
        self.view_max_frames = 30 # 判定結果の表示時間 30frame
        self.looking_view_frames = 0 # 注視判定結果の表示タイマー
        self.headtilt_view_frames = 0 # 首かしげ判定結果の表示タイマー

    def update_trackers(self, head_boxes: List[Box]):
        # 現在追跡中のオブジェクトの中心点を取得
        tracked_centroids: List[Tuple[int, Tuple[int, int]]] = [(t.id, (t.box.cx, t.box.cy)) for t in self.tracked_heads.values()]
        # 新しく検出されたバウンディングボックスの中心点を取得
        new_centroids: List[Tuple[int, int]] = [(box.cx, box.cy) for box in head_boxes]

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

    def match_objects(self, tracked_centroids:  List[Tuple[int, Tuple[int, int]]], new_centroids: List[Tuple[int, int]]):
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

    def stack_headtilt_history(self, tracked_id: int, state: bool):
        if tracked_id in self.tracked_heads:
            self.tracked_heads[tracked_id].headtilt_history_long.append(state)
            self.tracked_heads[tracked_id].headtilt_history_short.append(state)

    def get_looking_state_start(self, tracked_id: int) -> bool:
        if tracked_id in self.tracked_heads:
            state_interval, state_start, state_end = state_verdict(
                long_tracking_history=self.tracked_heads[tracked_id].looking_history_long,
                short_tracking_history=self.tracked_heads[tracked_id].looking_history_short,
            )
            return state_start
        return False

    def get_headtilt_state_start(self, tracked_id: int) -> bool:
        if tracked_id in self.tracked_heads:
            state_interval, state_start, state_end = state_verdict(
                long_tracking_history=self.tracked_heads[tracked_id].headtilt_history_long,
                short_tracking_history=self.tracked_heads[tracked_id].headtilt_history_short,
            )
            return state_start
        return False

    def get_looking_view_text(self, is_looking_start: bool) -> str:
        if is_looking_start:
            self.looking_view_frames = self.view_max_frames
            return 'Looking'
        else:
            if self.looking_view_frames > 0:
                self.looking_view_frames = self.looking_view_frames - 1
                return 'Looking'
            else:
                return ''

    def get_headtilt_view_text(self, is_headtilt_start: bool) -> str:
        if is_headtilt_start:
            self.headtilt_view_frames = self.view_max_frames
            return 'Tilt'
        else:
            if self.headtilt_view_frames > 0:
                self.headtilt_view_frames = self.headtilt_view_frames - 1
                return 'Tilt'
            else:
                return ''

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
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
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

            import onnx
            onnx_graph: onnx.ModelProto = onnx.load(model_path)
            if onnx_graph.graph.node[0].op_type == "Resize":
                first_resize_op: List[onnx.ValueInfoProto] = [i for i in onnx_graph.graph.value_info if i.name == "prep/Resize_output_0"]
                if first_resize_op:
                    self._input_shapes = [[d.dim_value for d in first_resize_op[0].type.tensor_type.shape.dim]]
                else:
                    self._input_shapes = [
                        input.shape for input in self._interpreter.get_inputs()
                    ]
            else:
                self._input_shapes = [
                    input.shape for input in self._interpreter.get_inputs()
                ]


            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class YOLOv9(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolov9_n_wholebody25_post_0100_1x3x480x640.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOv9. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOv9

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, atrributes, is_used=False]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]. [instances, [batchno, classid, score, x1, y1, x2, y2]].

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._obj_class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                # Object filter
                for box, score in zip(boxes_keep, scores_keep):
                    classid = int(box[1])
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    result_boxes.append(
                        Box(
                            classid=classid,
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            generation=-1, # -1: Unknown, 0: Adult, 1: Child
                            gender=-1, # -1: Unknown, 0: Male, 1: Female
                            handedness=-1, # -1: Unknown, 0: Left, 1: Right
                            head_pose=-1, # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
                        )
                    )
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [1,2,3,4,8,9,10,11,12,13,14,15] and box.score >= self._attr_class_score_th) or box.classid not in [1,2,3,4,8,9,10,11,12,13,14,15]
                ]

                # Adult, Child merge
                # classid: 0 -> Body
                #   classid: 1 -> Adult
                #   classid: 2 -> Child
                # 1. Calculate Adult and Child IoUs for Body detection results
                # 2. Connect either the Adult or the Child with the highest score and the highest IoU with the Body.
                # 3. Exclude Adult and Child from detection results
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]
                # Male, Female merge
                # classid: 0 -> Body
                #   classid: 3 -> Male
                #   classid: 4 -> Female
                # 1. Calculate Male and Female IoUs for Body detection results
                # 2. Connect either the Male or the Female with the highest score and the highest IoU with the Body.
                # 3. Exclude Male and Female from detection results
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]
                # HeadPose merge
                # classid: 7 -> Head
                #   classid:  8 -> Front
                #   classid:  9 -> Right-Front
                #   classid: 10 -> Right-Side
                #   classid: 11 -> Right-Back
                #   classid: 12 -> Back
                #   classid: 13 -> Left-Back
                #   classid: 14 -> Left-Side
                #   classid: 15 -> Left-Front
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8,9,10,11,12,13,14,15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8,9,10,11,12,13,14,15]]
                # Left and right hand merge
                # classid: 21 -> Hand
                #   classid: 22 -> Left-Hand
                #   classid: 23 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 21]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [22, 23]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [22, 23]]
        return result_boxes

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 22:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 23:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

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

def euler_to_rotation_matrix(yaw, pitch, roll, degrees=True):
    """
    オイラー角 (yaw, pitch, roll) から回転行列を生成する。
    - yaw: Y軸まわり回転
    - pitch: X軸まわり回転
    - roll: Z軸まわり回転
    - degrees=True なら度数法入力をラジアンに変換
    """
    if degrees:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    # Yaw 回転行列 (Y軸まわり)
    R_yaw = np.array([
        [ cy,  0, sy],
        [  0,  1,  0],
        [-sy,  0, cy]
    ], dtype=float)

    # Pitch 回転行列 (X軸まわり)
    R_pitch = np.array([
        [1,   0,    0 ],
        [0,  cp,  -sp ],
        [0,  sp,   cp ]
    ], dtype=float)

    # Roll 回転行列 (Z軸まわり)
    R_roll = np.array([
        [ cr, -sr,  0 ],
        [ sr,  cr,  0 ],
        [  0,   0,  1 ]
    ], dtype=float)

    # 最終的な回転行列 (回転順序に注意)
    # ここでは R = R_roll * R_pitch * R_yaw の順に掛け合わせています
    R = R_roll @ R_pitch @ R_yaw
    return R

def compute_head_tilt(yaw, pitch, roll, degrees=True) -> float:
    """
    オイラー角 (yaw, pitch, roll) から「頭部上方向ベクトル」と
    カメラ座標系の上方向ベクトル (0,1,0) がなす角度（度数）を返す。
    """
    R = euler_to_rotation_matrix(yaw, pitch, roll, degrees=degrees)

    # 頭部ローカル座標系での「上向きベクトル」
    u_local = np.array([0, 1, 0], dtype=float)

    # カメラ座標系に変換 (u_camera = R * u_local)
    u_camera = R @ u_local

    # カメラ座標系での「世界の上」ベクトル (0,1,0)
    v_up = np.array([0, 1, 0], dtype=float)

    # 2つのベクトルのなす角度 ( arccos((u・v)/(|u||v|)) )
    dot_val = np.dot(u_camera, v_up)
    norm_u = np.linalg.norm(u_camera)
    norm_v = np.linalg.norm(v_up)

    # arccos に与える値が浮動小数点誤差で [-1,1] を超えないようクリップする
    cos_val = dot_val / (norm_u * norm_v)
    cos_val = np.clip(cos_val, -1.0, 1.0)

    angle_rad = np.arccos(cos_val)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg

def is_headtilt_at_camera_with_angles(yaw, pitch, roll, tilt_threshold=15.0) -> bool:
    angle_deg = compute_head_tilt(yaw=yaw, pitch=pitch, roll=roll)
    if angle_deg > tilt_threshold:
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
        self.start_time = datetime.now().strftime("%Y%m%d%H%M%S")

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
        default='yolov9_n_wholebody25_post_0100_1x3x480x640.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cuda',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-it',
        '--inference_type',
        type=str,
        choices=['fp16', 'int8'],
        default='fp16',
        help='Inference type. Default: fp16',
    )
    parser.add_argument(
        '-ost',
        '--object_socre_threshold',
        type=float,
        default=0.35,
        help=\
            'The detection score threshold for object detection. Default: 0.35',
    )
    parser.add_argument(
        '-ast',
        '--attribute_socre_threshold',
        type=float,
        default=0.75,
        help=\
            'The attribute score threshold for object detection. Default: 0.70',
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

    # runtime check
    model_file: str = args.model
    model_dir_path = os.path.dirname(os.path.abspath(model_file))
    model_ext: str = os.path.splitext(model_file)[1][1:].lower()
    runtime: str = None
    if model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)
    video: str = args.video
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    execution_provider: str = args.execution_provider
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()
    providers: List[Tuple[str, Dict] | str] = None

    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        elif inference_type == 'int8':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                    "trt_int8_enable": True,
                    "trt_int8_calibration_table_name": "calibration.flatbuffers",
                }
        else:
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': f'{model_dir_path}',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                } | ep_type_params,
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]

    enable_log: bool = args.enable_log
    max_logging_instances: int = args.max_logging_instances
    max_logging_rows: int = args.max_logging_rows
    looking_duration: int = args.looking_duration

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    model = YOLOv9(
        runtime=runtime,
        model_path=model_file,
        obj_class_score_th=object_socre_threshold,
        attr_class_score_th=attribute_socre_threshold,
        providers=providers,
    )

    session_option = onnxruntime.SessionOptions()
    session_option.log_severity_level = 3
    onnx_session = onnxruntime.InferenceSession(
        'sixdrepnet360_Nx3x224x224_full.onnx',
        sess_options=session_option,
        providers=providers,
    )

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    cap = cv2.VideoCapture(
        int(video) if is_parsable_to_int(video) else video
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

    head_tracker = HeadTracker(max_distance=50, max_lost=30, looking_duration=int(looking_duration * cap_fps))

    log_writer: LogWriter = None
    NUMBER_OF_COLUMNS_PER_LOG_PER_INSTANCE: int = 3
    if enable_log:
        header_row = ['timestamp', 'frameno']
        for idx in range(max_logging_instances):
            header_row += [f'trackid_{idx+1}',  f'looking_{idx+1}', f'headtilt_{idx+1}',]
        log_writer = LogWriter(base_filename="log.csv", max_lines=max_logging_rows, header_row=header_row)

    frame_no: int = 0 # ログ出力用動画フレーム番号

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)

        start_time = time.time()
        boxes = model(
            image=debug_image,
            disable_generation_identification_mode=True,
            disable_gender_identification_mode=True,
            disable_left_and_right_hand_identification_mode=True,
            disable_headpose_identification_mode=True,
        )

        boxes = [box for box in boxes if box.classid == 7] # Head

        # ログ出力用タイムスタンプ取得
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S") + f'{now.microsecond // 1000:03d}'

        log_row = [timestamp, frame_no]
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

                is_looking = is_looking_at_camera_with_angles(tracked_head.box, yaw_deg, pitch_deg, image_width, image_height)
                head_tracker.stack_looking_history(tracked_head.id, is_looking)
                is_looking_start = head_tracker.get_looking_state_start(tracked_head.id)

                is_headtilt = is_headtilt_at_camera_with_angles(yaw=yaw_deg, pitch=pitch_deg, roll=roll_deg, tilt_threshold=12.0)
                head_tracker.stack_headtilt_history(tracked_head.id, is_headtilt)
                is_headtilt_start = head_tracker.get_headtilt_state_start(tracked_head.id)

                looking_camera_txt = head_tracker.get_looking_view_text(is_looking_start)
                head_tilt_txt = head_tracker.get_headtilt_view_text(is_headtilt_start)

                cv2.putText(
                    debug_image,
                    f'{tracked_head.id:06} {looking_camera_txt} {head_tilt_txt}',
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
                    f'{tracked_head.id:06} {looking_camera_txt} {head_tilt_txt}',
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

                # ログ出力
                if enable_log:
                    log_row.append(f'{tracked_head.id}')
                    log_row.append(f'{is_looking_start}')
                    log_row.append(f'{is_headtilt_start}')

            # timestamp, frame_no, tracked_head.id, is_looking_start, is_headtilt_start, ... (tracked_head.id〜is_headtilt_start x20)
            if enable_log:
                while len(log_row) < (max_logging_instances * NUMBER_OF_COLUMNS_PER_LOG_PER_INSTANCE + 2):
                    log_row.append('')
                log_writer.write_row(log_row)

        else:
            # ログ出力
            if enable_log:
                log_row = []
                log_row.append(timestamp)
                # timestamp, frame_no, tracked_head.id, is_looking_start, is_headtilt_start, ... (tracked_head.id〜is_headtilt_start x20)
                while len(log_row) < (max_logging_instances * NUMBER_OF_COLUMNS_PER_LOG_PER_INSTANCE + 2):
                    log_row.append('')
                log_writer.write_row(log_row)

        frame_no += 1 # ログ出力用フレーム番号

        elapsed_time = time.time() - start_time
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

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
