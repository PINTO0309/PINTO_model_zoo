#!/usr/bin/env python

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import cv2
import time
from pprint import pprint
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod
import onnx
import ast

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
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    is_hand_used: bool = False
    race: str = ''
    gender: str = ''
    age: str = ''
    emotion: str = ''

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _class_score_th: float = 0.35
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

    _mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    _std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
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
        self._runtime = runtime
        self._model_path = model_path
        self._class_score_th = class_score_th
        self._providers = providers
        self.onnx_graph = None

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

            self.onnx_graph: onnx.ModelProto = onnx.load(model_path)
            if self.onnx_graph.graph.node[0].op_type == "Resize":
                first_resize_op: List[onnx.ValueInfoProto] = [i for i in self.onnx_graph.graph.value_info if i.name == "prep/Resize_output_0"]
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
        model_path: Optional[str] = 'yolov9_t_wholebody_with_wheelchair_post_0100_1x3x384x672.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOv9. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOv9

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
        disable_left_and_right_hand_discrimination_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_left_and_right_hand_discrimination_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, handedness, is_hand_used=False]
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
                disable_left_and_right_hand_discrimination_mode=disable_left_and_right_hand_discrimination_mode,
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
        disable_left_and_right_hand_discrimination_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        disable_left_and_right_hand_discrimination_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, handedness, is_hand_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
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
                            handedness=-1 if classid not in [5, 6] else classid - 5, # -1: None, 0: Left, 1: Right
                        )
                    )
                # Left and right hand merge
                # classid: 4 -> Hand
                #   classid: 5 -> Left-Hand
                #   classid: 6 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_discrimination_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 4]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [5, 6]]
                    self._find_most_relevant_hand(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [5, 6]]

        return result_boxes

    def _find_most_relevant_hand(
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
                if target_obj is not None and not target_obj.is_hand_used:
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
                            best_distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 5:
                    base_obj.handedness = 0
                    most_relevant_obj.is_hand_used = True
                elif most_relevant_obj.classid == 6:
                    base_obj.handedness = 1
                    most_relevant_obj.is_hand_used = True
                else:
                    base_obj.handedness = -1

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

class FairDAN(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'fairdan_affectnet8_Nx3x224x224.onnx',
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for FairDAN. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for FairDAN

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        if self._runtime == 'onnx':
            self._swap = (0,3,1,2)
        else:
            self._swap = (0,1,2,3)
        self.meta_datas = {}
        if self.onnx_graph is not None and self.onnx_graph.metadata_props is not None:
            meta_datas = {metadata_prop.key: metadata_prop.value for metadata_prop in self.onnx_graph.metadata_props}
        if meta_datas:
            self._channel_order: str = meta_datas.get('channel_order', 'rgb')
            self._mean: np.ndarray = np.asarray(ast.literal_eval(meta_datas.get('mean', self._mean)), dtype=np.float32)
            self._mean = self._mean.reshape([3,1,1])
            self._std: np.ndarray = np.asarray(ast.literal_eval(meta_datas.get('std', self._std)), dtype=np.float32)
            self._std = self._std.reshape([3,1,1])
            self.gender_labels: List[str] = ast.literal_eval(meta_datas.get('gender_labels', ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']))
            self.race_labels: List[str] = ast.literal_eval(meta_datas.get('race_labels', ['0','1','2','3','4','5','6']))
            self.age_labels: List[str] = ast.literal_eval(meta_datas.get('age_labels', ['0','1','2','3','4','5','6','7','8']))
            self.emotion_labels: List[str] = ast.literal_eval(meta_datas.get('emotion_labels', ['0','1','2','3','4','5','6','7','8']))

    def __call__(
        self,
        images: List[np.ndarray],
    ) -> List[List[str, str, str]]:
        """

        Parameters
        ----------
        images: List[np.ndarray]
            Face images

        Returns
        -------
        result_labels: List[List[str, str, str, str]]
            raceid, genderid, ageid, emotionid
        """
        # PreProcess
        normalized_images = \
            self._preprocess(
                images,
            )
        # Inference
        inferece_image = np.asarray(normalized_images, dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        # [N, 4]
        raceids_genderids_ageids_emotionids = outputs[0]
        result_labels = self._postprocess(raceids_genderids_ageids_emotionids=raceids_genderids_ageids_emotionids)
        return result_labels

    def _preprocess(
        self,
        images: List[np.ndarray],
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple

        Returns
        -------
        resized_images: np.ndarray
            Resized and normalized image.
        """
        preped_images: List[np.ndarray] = []
        for image in images:
            preped_images.append(
                cv2.resize(
                    image,
                    (self._input_shapes[0][self._h_index] , self._input_shapes[0][self._w_index])
                )
            )
        np_preped_images = np.asarray(preped_images)
        if self._channel_order == 'rgb':
            np_preped_images = np_preped_images[..., ::-1] # BGR -> RGB
        np_preped_images = np_preped_images.transpose(self._swap)
        np_preped_images = (np_preped_images / 255.0 - self._mean) / self._std
        return np_preped_images

    def _postprocess(
        self,
        *,
        raceids_genderids_ageids_emotionids: np.ndarray,
    ) -> List[List[str, str, str, str]]:
        result_labels = []
        for raceid_genderid_ageid_emotionid in raceids_genderids_ageids_emotionids:
            result_labels.append(
                [
                    self.race_labels[raceid_genderid_ageid_emotionid[0]],
                    self.gender_labels[raceid_genderid_ageid_emotionid[1]],
                    self.age_labels[raceid_genderid_ageid_emotionid[2]],
                    self.emotion_labels[raceid_genderid_ageid_emotionid[3]],
                ]
            )
        return result_labels

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
    parser.add_argument(
        '-dm',
        '--detection_model',
        type=str,
        default='yolov9_s_wholebody_with_wheelchair_post_0100_1x3x480x640.onnx',
        help='ONNX/TFLite file path for YOLOv9.',
    )
    parser.add_argument(
        '-am',
        '--attributes_model',
        type=str,
        default='fairdan_affectnet8_Nx3x224x224.onnx',
        help='ONNX/TFLite file path for DAN.',
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
        '-dlr',
        '--disable_left_and_right_hand_discrimination_mode',
        action='store_true',
        help=\
            'Disable left and right hand discrimination mode.',
    )
    args = parser.parse_args()

    detection_model_file: str = args.detection_model
    model_dir_path = os.path.dirname(os.path.abspath(detection_model_file))
    detection_model_ext: str = os.path.splitext(detection_model_file)[1][1:].lower()

    attributes_model_file: str = args.attributes_model
    attributes_model_ext: str = os.path.splitext(attributes_model_file)[1][1:].lower()

    assert detection_model_ext == attributes_model_ext, "Model extensions should be consistent. .onnx or .tflite"

    runtime: str = None
    if detection_model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif detection_model_ext == 'tflite':
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
    disable_left_and_right_hand_discrimination_mode: bool = args.disable_left_and_right_hand_discrimination_mode
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

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    # Model initialization
    detection_model = YOLOv9(
        runtime=runtime,
        model_path=detection_model_file,
        class_score_th=0.35,
        providers=providers,
    )
    attributes_model = FairDAN(
        runtime=runtime,
        model_path=attributes_model_file,
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

        debug_image = copy.deepcopy(image)
        # debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        boxes = detection_model(
            image=debug_image,
            disable_left_and_right_hand_discrimination_mode=disable_left_and_right_hand_discrimination_mode,
        )

        # Face only filter
        face_boxes = [box for box in boxes if box.classid == 3]

        if len(face_boxes) > 0:
            attributes: List[List[str, str, str]] = \
                attributes_model(
                    images=[debug_image[face_box.y1:face_box.y2, face_box.x1: face_box.x2, :] for face_box in face_boxes],
                )
            for face_box, attribute in zip(face_boxes, attributes):
                race, gender, age, emotion = attribute
                face_box.race = race
                face_box.gender = gender
                face_box.age = age
                face_box.emotion = emotion

        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)
            if classid == 0:
                # Body
                color = (255,0,0)
            elif classid == 1:
                # Body-With-Wheelchair
                color = (0,200,255)
            elif classid == 2:
                # Head
                color = (0,0,255)
            elif classid == 3:
                # Face
                color = (0,200,255)
            elif classid == 4:
                if not disable_left_and_right_hand_discrimination_mode:
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
            elif classid == 7:
                # Foot
                color = (0,0,255)

            if classid not in [1, 3]:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 3)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                if not disable_left_and_right_hand_discrimination_mode and classid == 4:
                    handedness_txt = 'Left' if box.handedness == 0 else 'Right' if box.handedness == 1 else 'Unknown'
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
            else:
                draw_dashed_rectangle(
                    image=debug_image,
                    top_left=(box.x1, box.y1),
                    bottom_right=(box.x2, box.y2),
                    color=color,
                    thickness=2,
                    dash_length=10
                )

                cv2.rectangle(
                    debug_image,
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-115 if box.y1-130 > 0 else 20,
                    ),
                    (
                        box.x1+200 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-5,
                    ),
                    (255,240,240),
                    cv2.FILLED,
                )

                cv2.putText(
                    debug_image,
                    f'{box.race}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-90 if box.y1-105 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{box.race}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-90 if box.y1-105 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug_image,
                    f'{box.gender}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-65 if box.y1-80 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{box.gender}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-65 if box.y1-80 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug_image,
                    f'{box.age}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-40 if box.y1-55 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{box.age}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-40 if box.y1-55 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug_image,
                    f'{box.emotion}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-15 if box.y1-30 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{box.emotion}',
                    (
                        box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                        box.y1-15 if box.y1-30 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,0),
                    1,
                    cv2.LINE_AA,
                )

        if file_paths is not None:
            basename = os.path.basename(file_paths[file_paths_count])
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{basename}', debug_image)
        if video_writer is not None:
            video_writer.write(debug_image)

        cv2.imshow("test", debug_image)

        key = cv2.waitKey(1) if file_paths is None or disable_waitKey else cv2.waitKey(0)
        if key == 27: # ESC
            break

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
