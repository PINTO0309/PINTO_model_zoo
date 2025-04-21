#!/usr/bin/env python

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import cv2
try:
    import onnx
    import onnxruntime
    from sne4onnx import extraction
    from sor4onnx import rename
except:
    pass
import time
from pprint import pprint
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentTypeError
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod

BOX_COLORS = [
    [(216, 67, 21),"Front"],
    [(255, 87, 34),"Right-Front"],
    [(123, 31, 162),"Right-Side"],
    [(255, 193, 7),"Right-Back"],
    [(76, 175, 80),"Back"],
    [(33, 150, 243),"Left-Back"],
    [(156, 39, 176),"Left-Side"],
    [(0, 188, 212),"Left-Front"],
]

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

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70

    _input_shapes: List[List[int | str]] = []
    _input_names: List[str] = []
    _input_dtypes: List[np.dtype] = []
    _output_shapes: List[List[int | str]] = []
    _output_names: List[str] = []

    _input_shapes_postprocess: List[List[int | str]] = []
    _input_names_postprocess: List[str] = []
    _input_dtypes_postprocess: List[np.dtype] = []
    _output_shapes_postprocess: List[List[int | str]] = []
    _output_names_postprocess: List[str] = []

    # onnx/tflite
    _interpreter = None
    _interpreter_postprocess = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes: Dict[str, np.dtype] = {
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
        model_path_post: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
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
        self._model_path_post = model_path_post
        self._model = None
        self._model_postprocess = None
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3

            # Initialize model body
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

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

            # Support for TensorRT 9.x+
            # Initialize model post-process
            if any((p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider" for p in providers) and model_path_post:
                self._interpreter_postprocess = \
                    onnxruntime.InferenceSession(
                        model_path_post,
                        sess_options=session_option,
                        providers=['CPUExecutionProvider'],
                    )
                self._input_names_postprocess = [
                    input.name for input in self._interpreter_postprocess.get_inputs()
                ]
                self._input_dtypes_postprocess = [
                    self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter_postprocess.get_inputs()
                ]
                self._output_shapes_postprocess = [
                    output.shape for output in self._interpreter_postprocess.get_outputs()
                ]
                self._output_names_postprocess = [
                    output.name for output in self._interpreter_postprocess.get_outputs()
                ]
                self._model_postprocess = self._interpreter_postprocess.run

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
            # Support for TensorRT 9.x+
            # Isolation of NMS
            if self._model_postprocess:
                datas = {
                    f'{input_name}': input_data \
                        for input_name, input_data in zip(self._output_names, outputs)
                }
                outputs = [
                    output for output in \
                        self._model_postprocess(
                            output_names=self._output_names_postprocess,
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

    # Support for TensorRT 9.x+
    def model_split(
        self,
        *,
        model_path: str,
        output_model_path: str,
        runtime: Optional[str] = 'onnx',
        input_op_names: List[str] = ['input'],
        output_op_names: List[str] = ['output'],
    ) -> onnx.ModelProto:
        """https://github.com/PINTO0309/sne4onnx

        Parameters
        ----------
        model_path: str
            ONNX file path for YOLOv9

        output_model_path: str
            ONNX file path for YOLOv9

        runtime: Optional[str]
            Default: 'onnx'

        input_op_names: List[str]
            Default: ['input']

        output_op_names: List[str]
            Default: ['output']

        Returns
        -------
        extracted_model: onnx.ModelProto
        """
        if runtime != 'onnx':
            raise NotImplementedError()
        extracted_model = extraction(
            input_op_names=input_op_names,
            output_op_names=output_op_names,
            input_onnx_file_path=model_path,
            output_onnx_file_path=output_model_path,
            non_verbose=True,
        )
        return extracted_model

    # Support for TensorRT 9.x+
    def model_op_rename(
        self,
        *,
        model_path: str,
        output_model_path: str,
        runtime: Optional[str] = 'onnx',
        old_new: List[str] = ['input', 'input'],
        mode: str = 'full',
        search_mode: str = 'exact_match',
    ) -> onnx.ModelProto:
        """https://github.com/PINTO0309/sor4onnx

        Parameters
        ----------
        model_path: str
            ONNX file path for YOLOv9

        output_model_path: str
            ONNX file path for YOLOv9

        runtime: Optional[str]
            Default: 'onnx'

        old_new: List[str]
            Default: ['input', 'input']

        mode: str
            Default: 'full'

        search_mode: str
            Default: 'exact_match'

        Returns
        -------
        renamed_model: onnx.ModelProto
        """
        if runtime != 'onnx':
            raise NotImplementedError()
        renamed_model = rename(
            old_new=old_new,
            input_onnx_file_path=model_path,
            output_onnx_file_path=output_model_path,
            non_verbose=True,
        )
        return renamed_model

class YOLOv9(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolov9_e_wholebody28_refine_post_0100_1x3x480x640.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
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

        keypoint_th: Optional[float]
            Keypoints score threshold. Default: 0.25

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        # Support for TensorRT 9.x+
        splited_model_body_path = model_path
        splited_model_post_path = ''
        if runtime == 'onnx' and any((p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider" for p in providers):
            # Support for TensorRT 9.x+, Isolation of NMS

            # Model body part generation
            splited_model_body_path = f"{os.path.splitext(os.path.basename(model_path))[0]}_body.onnx"
            if not os.path.isfile(splited_model_body_path):
                super().model_split(
                    model_path=model_path,
                    output_model_path=splited_model_body_path,
                    runtime=runtime,
                    input_op_names=['input_bgr'],
                    output_op_names=['x1y1x2y2', 'main01_y1x1y2x2', 'main01_scores'],
                )
                super().model_op_rename(
                    model_path=splited_model_body_path,
                    output_model_path=splited_model_body_path,
                    runtime=runtime,
                    old_new=['main01_scores', 'scores'],
                    mode='outputs',
                    search_mode='exact_match',
                )
                super().model_op_rename(
                    model_path=splited_model_body_path,
                    output_model_path=splited_model_body_path,
                    runtime=runtime,
                    old_new=['main01_y1x1y2x2', 'y1x1y2x2'],
                    mode='outputs',
                    search_mode='exact_match',
                )

            # Model post-process part generation
            splited_model_post_path = f"{os.path.splitext(os.path.basename(model_path))[0]}_post.onnx"
            if not os.path.isfile(splited_model_post_path):
                super().model_split(
                    model_path=model_path,
                    output_model_path=splited_model_post_path,
                    runtime=runtime,
                    input_op_names=['x1y1x2y2', 'main01_y1x1y2x2', 'main01_scores'],
                    output_op_names=['batchno_classid_score_x1y1x2y2'],
                )
                super().model_op_rename(
                    model_path=splited_model_post_path,
                    output_model_path=splited_model_post_path,
                    runtime=runtime,
                    old_new=['main01_scores', 'scores'],
                    mode='inputs',
                    search_mode='exact_match',
                )
                super().model_op_rename(
                    model_path=splited_model_post_path,
                    output_model_path=splited_model_post_path,
                    runtime=runtime,
                    old_new=['main01_y1x1y2x2', 'y1x1y2x2'],
                    mode='inputs',
                    search_mode='exact_match',
                )

        super().__init__(
            runtime=runtime,
            model_path=splited_model_body_path,
            model_path_post=splited_model_post_path if splited_model_post_path else '',
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            keypoint_th=keypoint_th,
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

        box_score_threshold: float = min([self._obj_class_score_th, self._attr_class_score_th, self._keypoint_th])

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > box_score_threshold
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
                # Object filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [0,5,6,7,16,17,18,19,20,23,24,25,27] and box.score >= self._obj_class_score_th) or box.classid not in [0,5,6,7,16,17,18,19,20,23,24,25,27]
                ]
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [1,2,3,4,8,9,10,11,12,13,14,15] and box.score >= self._attr_class_score_th) or box.classid not in [1,2,3,4,8,9,10,11,12,13,14,15]
                ]
                # Keypoint filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [21,22,26] and box.score >= self._keypoint_th) or box.classid not in [21,22,26]
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
                # classid: 23 -> Hand
                #   classid: 24 -> Left-Hand
                #   classid: 25 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 23]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [24, 25]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [24, 25]]
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

                elif most_relevant_obj.classid == 24:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 25:
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

def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted([str(file) for file in image_files])

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

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)


def main():
    parser = ArgumentParser()

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 2:
            raise ArgumentTypeError(f"Invalid Value: {ivalue}. Please specify an integer of 2 or greater.")
        return ivalue

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='yolov9_e_wholebody28_refine_post_0100_1x3x480x640.onnx',
        help='ONNX/TFLite file path for YOLOv9.',
    )
    group_v_or_i = parser.add_mutually_exclusive_group(required=True)
    group_v_or_i.add_argument(
        '-v',
        '--video',
        type=str,
        help='Video file path or camera index.',
    )
    group_v_or_i.add_argument(
        '-i',
        '--images_dir',
        type=str,
        help='jpg, png images folder path.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cpu',
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
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    parser.add_argument(
        '-dwk',
        '--disable_waitKey',
        action='store_true',
        help=\
            'Disable cv2.waitKey(). '+
            'When you want to process a batch of still images, '+
            ' disable key-input wait and process them continuously.',
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
        '-kst',
        '--keypoint_threshold',
        type=float,
        default=0.25,
        help=\
            'The keypoint score threshold for object detection. Default: 0.25',
    )
    parser.add_argument(
        '-kdm',
        '--keypoint_drawing_mode',
        type=str,
        choices=['dot', 'box', 'both'],
        default='dot',
        help='Key Point Drawing Mode. Default: dot',
    )
    parser.add_argument(
        '-dnm',
        '--disable_generation_identification_mode',
        action='store_true',
        help=\
            'Disable generation identification mode. (Press N on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dgm',
        '--disable_gender_identification_mode',
        action='store_true',
        help=\
            'Disable gender identification mode. (Press G on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dlr',
        '--disable_left_and_right_hand_identification_mode',
        action='store_true',
        help=\
            'Disable left and right hand identification mode. (Press H on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhm',
        '--disable_headpose_identification_mode',
        action='store_true',
        help=\
            'Disable HeadPose identification mode. (Press P on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-drc',
        '--disable_render_classids',
        type=int,
        nargs="*",
        default=[],
        help=\
            'Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19',
    )
    parser.add_argument(
        '-efm',
        '--enable_face_mosaic',
        action='store_true',
        help=\
            'Enable face mosaic.',
    )
    parser.add_argument(
        '-oyt',
        '--output_yolo_format_text',
        action='store_true',
        help=\
            'Output YOLO format texts and images.',
    )
    parser.add_argument(
        '-bblw',
        '--bounding_box_line_width',
        type=check_positive,
        default=2,
        help=\
            'Bounding box line width. Default: 2',
    )
    args = parser.parse_args()

    # runtime check
    model_file: str = args.model
    model_dir_path = os.path.dirname(os.path.abspath(model_file))
    model_ext: str = os.path.splitext(model_file)[1][1:].lower()
    runtime: str = None
    execution_provider: str = args.execution_provider
    if model_ext == 'onnx':
        err_msg = ''
        if not is_package_installed('onnx') or \
            not is_package_installed('onnxruntime'):
            err_msg = f'onnx onnxruntime'
        if execution_provider == 'tensorrt' and \
            (
                not is_package_installed('sne4onnx') or \
                not is_package_installed('sor4onnx') or \
                not is_package_installed('onnx_graphsurgeon')
            ):
            err_msg = f'{err_msg} sne4onnx sor4onnx onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com'.lstrip()
        if err_msg:
            print(Color.RED(f'ERROR: {err_msg} is not installed. pip install {err_msg}'))
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
    images_dir: str = args.images_dir
    disable_waitKey: bool = args.disable_waitKey
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    keypoint_threshold: float = args.keypoint_threshold
    keypoint_drawing_mode: str = args.keypoint_drawing_mode
    disable_generation_identification_mode: bool = args.disable_generation_identification_mode
    disable_gender_identification_mode: bool = args.disable_gender_identification_mode
    disable_left_and_right_hand_identification_mode: bool = args.disable_left_and_right_hand_identification_mode
    disable_headpose_identification_mode: bool = args.disable_headpose_identification_mode
    disable_render_classids: List[int] = args.disable_render_classids
    enable_face_mosaic: bool = args.enable_face_mosaic
    output_yolo_format_text: bool = args.output_yolo_format_text
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()
    bounding_box_line_width: int = args.bounding_box_line_width
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

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    # Model initialization
    model = YOLOv9(
        runtime=runtime,
        model_path=model_file,
        obj_class_score_th=object_socre_threshold,
        attr_class_score_th=attribute_socre_threshold,
        keypoint_th=keypoint_threshold,
        providers=providers,
    )

    file_paths: List[str] = None
    cap = None
    video_writer = None
    if images_dir is not None:
        file_paths = list_image_files(dir_path=images_dir)
    else:
        cap = cv2.VideoCapture(
            int(video) if is_parsable_to_int(video) else video
        )
        disable_video_writer: bool = args.disable_video_writer
        if not disable_video_writer:
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                filename='output.mp4',
                fourcc=fourcc,
                fps=cap_fps,
                frameSize=(w, h),
            )

    file_paths_count = -1
    movie_frame_count = 0
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1
    while True:
        image: np.ndarray = None
        if file_paths is not None:
            file_paths_count += 1
            if file_paths_count <= len(file_paths) - 1:
                image = cv2.imread(file_paths[file_paths_count])
            else:
                break
        else:
            res, image = cap.read()
            if not res:
                break
            movie_frame_count += 1

        debug_image = copy.deepcopy(image)
        debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        boxes = model(
            image=debug_image,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        elapsed_time = time.perf_counter() - start_time

        if file_paths is None:
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw bounding boxes
        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)

            if classid in disable_render_classids:
                continue

            if classid == 0:
                # Body
                if not disable_gender_identification_mode:
                    # Body
                    if box.gender == 0:
                        # Male
                        color = (255,0,0)
                    elif box.gender == 1:
                        # Female
                        color = (139,116,225)
                    else:
                        # Unknown
                        color = (0,200,255)
                else:
                    # Body
                    color = (0,200,255)
            elif classid == 5:
                # Body-With-Wheelchair
                color = (0,200,255)
            elif classid == 6:
                # Body-With-Crutches
                color = (83,36,179)
            elif classid == 7:
                # Head
                if not disable_headpose_identification_mode:
                    color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216,67,21)
                else:
                    color = (0,0,255)
            elif classid == 16:
                # Face
                color = (0,200,255)
            elif classid == 17:
                # Eye
                color = (255,0,0)
            elif classid == 18:
                # Nose
                color = (0,255,0)
            elif classid == 19:
                # Mouth
                color = (0,0,255)
            elif classid == 20:
                # Ear
                color = (203,192,255)
            elif classid == 21:
                # Shoulder
                color = (255,0,0)
            elif classid == 22:
                # Elbow
                color = (0,255,0)
            elif classid == 23:
                if not disable_left_and_right_hand_identification_mode:
                    # Hands
                    if box.handedness == 0:
                        # Left-Hand
                        color = (0,128,0)
                    elif box.handedness == 1:
                        # Right-Hand
                        color = (255,0,255)
                    else:
                        # Unknown
                        color = (0,255,0)
                else:
                    # Hands
                    color = (0,255,0)
            elif classid == 26:
                # Knee
                color = (0,0,255)
            elif classid == 27:
                # Foot
                color = (250,0,136)

            if (classid == 0 and not disable_gender_identification_mode) \
                or (classid == 7 and not disable_headpose_identification_mode) \
                or (classid == 23 and not disable_left_and_right_hand_identification_mode) \
                or classid == 16 \
                or classid in [21, 22, 26]:

                # Body
                if classid == 0:
                    if box.gender == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Head
                elif classid == 7:
                    if box.head_pose == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Face
                elif classid == 16:
                    if enable_face_mosaic:
                        w = int(abs(box.x2 - box.x1))
                        h = int(abs(box.y2 - box.y1))
                        small_box = cv2.resize(debug_image[box.y1:box.y2, box.x1:box.x2, :], (3,3))
                        normal_box = cv2.resize(small_box, (w,h))
                        if normal_box.shape[0] != abs(box.y2 - box.y1) \
                            or normal_box.shape[1] != abs(box.x2 - box.x1):
                                normal_box = cv2.resize(small_box, (abs(box.x2 - box.x1), abs(box.y2 - box.y1)))
                        debug_image[box.y1:box.y2, box.x1:box.x2, :] = normal_box
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Hands
                elif classid == 23:
                    if box.handedness == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Shoulder, Elbow, Knee
                elif classid in [21, 22, 26]:
                    if keypoint_drawing_mode in ['dot', 'both']:
                        cv2.circle(debug_image, (box.cx, box.cy), 5, (255,255,255), -1)
                        cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
                    if keypoint_drawing_mode in ['box', 'both']:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)

            else:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

            # Attributes text
            generation_txt = ''
            if box.generation == -1:
                generation_txt = ''
            elif box.generation == 0:
                generation_txt = 'Adult'
            elif box.generation == 1:
                generation_txt = 'Child'

            gender_txt = ''
            if box.gender == -1:
                gender_txt = ''
            elif box.gender == 0:
                gender_txt = 'M'
            elif box.gender == 1:
                gender_txt = 'F'

            attr_txt = f'{generation_txt}({gender_txt})' if gender_txt != '' else f'{generation_txt}'

            headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
            attr_txt = f'{attr_txt} {headpose_txt}' if headpose_txt != '' else f'{attr_txt}'

            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            handedness_txt = ''
            if box.handedness == -1:
                handedness_txt = ''
            elif box.handedness == 0:
                handedness_txt = 'L'
            elif box.handedness == 1:
                handedness_txt = 'R'
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )

        if file_paths is not None:
            basename = os.path.basename(file_paths[file_paths_count])
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{basename}', debug_image)

        if file_paths is not None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_i.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_o.png', debug_image)
            with open(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        elif file_paths is None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{movie_frame_count:08d}.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_i.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_o.png', debug_image)
            with open(f'output/{movie_frame_count:08d}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        if video_writer is not None:
            video_writer.write(debug_image)
            # video_writer.write(image)

        cv2.imshow("test", debug_image)

        key = cv2.waitKey(1) if file_paths is None or disable_waitKey else cv2.waitKey(0)
        if key == 27: # ESC
            break
        elif key == 110: # N, Generation mode switch
            disable_generation_identification_mode = not disable_generation_identification_mode
        elif key == 103: # G, Gender mode switch
            disable_gender_identification_mode = not disable_gender_identification_mode
        elif key == 112: # P, HeadPose mode switch
            disable_headpose_identification_mode = not disable_headpose_identification_mode
        elif key == 104: # H, HandsLR mode switch
            disable_left_and_right_hand_identification_mode = not disable_left_and_right_hand_identification_mode
        elif key == 107: # K, Keypoints mode switch
            if keypoint_drawing_mode == 'dot':
                keypoint_drawing_mode = 'box'
            elif keypoint_drawing_mode == 'box':
                keypoint_drawing_mode = 'both'
            elif keypoint_drawing_mode == 'both':
                keypoint_drawing_mode = 'dot'

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
