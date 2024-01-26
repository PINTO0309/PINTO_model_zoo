#!/usr/bin/env python

import copy
import cv2
import time
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict

BODY_COLORS = [
    (110, 64, 170), (143, 61, 178), (178, 60, 178), (210, 62, 167),
    (238, 67, 149), (255, 78, 125), (255, 94, 99),  (255, 115, 75),
    (255, 140, 56), (239, 167, 47), (217, 194, 49), (194, 219, 64),
    (175, 240, 91), (135, 245, 87), (96, 247, 96),  (64, 243, 115),
    (40, 234, 141), (28, 219, 169), (26, 199, 194), (33, 176, 213),
    (47, 150, 224), (65, 125, 224), (84, 101, 214), (99, 81, 195)
]

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

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

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
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
            self.strides: int = 0

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
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

        elif self._runtime == 'openvino':
            import openvino as ov  # type: ignore

            core = ov.Core()
            model = core.read_model(model=model_path)

            compiled_model = core.compile_model(model=model, device_name="AUTO")

            self._interpreter = compiled_model
            # self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                list(input.shape) for input in self._interpreter.inputs
            ]
            self._input_names = [
                input.node.friendly_name for input in self._interpreter.inputs
            ]
            self._input_dtypes = [
                input.element_type.to_dtype().type for input in self._interpreter.inputs
            ]
            self._output_shapes = [
                list(output.shape) for output in self._interpreter.outputs
            ]
            self._output_names = [
                output.node.friendly_name for output in self._interpreter.outputs
            ]
            self._model = compiled_model
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3
            self.strides: int = 0

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

        elif self._runtime == 'openvino':
            infer_request = self._model.create_infer_request()

            infer_request.infer(inputs=datas)

            infer_request.start_async()
            infer_request.wait()

            outputs = [infer_request.get_output_tensor(i).data for i in range(len(self._output_names))]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        pass

    @property
    def input_shapes(self) -> List[List[int]]:
        return self._input_shapes

    @property
    def input_size(self) -> Tuple[int, int]:
        shape = self.input_shapes[0]
        return shape[self._w_index], shape[self._h_index]


class BodyPix(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'bodypix_resnet50_stride16_1x3x480x640.onnx',
        providers: Optional[List] = None,
        strides: Optional[int] = None
    ):
        """BodyPix

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for BodyPix. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for BodyPix

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self._swap = (2,0,1)
        self._mean = np.asarray([0.0, 0.0, 0.0])
        self._std = np.asarray([1.0, 1.0, 1.0])
        self.strides = strides

        # find strides
        if self.strides is None:
            import onnx
            model_proto = onnx.load(f=model_path)
            float_segments_raw_output = [v for v in model_proto.graph.value_info if v.name == 'float_segments_raw_output']
            if len(float_segments_raw_output) >= 1:
                w = float_segments_raw_output[0].type.tensor_type.shape.dim[-1].dim_value
                self.strides = self._input_shapes[0][self._w_index] // w

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        foreground_mask_zero_or_255: np.ndarray
            Predicted foreground mask: [batch, H, W, 3]. 0 or 255

        colored_mask_classid: np.ndarray
            Predicted colored mask: [batch, H, W, 1]. 0 - 24

        keypoints_classidscorexy: np.ndarray
            Predicted keypoints: [batch, N, 4]. classid, score, x, y
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
        foreground_mask_zero_or_255 = outputs[0]
        colored_mask_classid = outputs[1]
        keypoints_classidscorexy = outputs[2]
        # PostProcess
        result_foreground_mask_zero_or_255, result_colored_mask_classid, result_keypoints_classidscorexy = \
            self._postprocess(
                foreground_mask_zero_or_255=foreground_mask_zero_or_255,
                colored_mask_classid=colored_mask_classid,
                keypoints_classidscorexy=keypoints_classidscorexy,
            )
        return result_foreground_mask_zero_or_255, result_colored_mask_classid, result_keypoints_classidscorexy

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
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(self._swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )
        return resized_image

    def _postprocess(
        self,
        foreground_mask_zero_or_255: np.ndarray,
        colored_mask_classid: np.ndarray,
        keypoints_classidscorexy: np.ndarray,
    ) -> np.ndarray:
        """_postprocess

        Parameters
        ----------
        foreground_mask_zero_or_255: np.ndarray
            Predicted foreground mask: [batch, H, W, 3]. 0 or 255

        colored_mask_classid: np.ndarray
            Predicted colored mask: [batch, H, W, 1]. 0 - 24

        keypoints_classidscorexy: np.ndarray
            Predicted keypoints: [batch, N, 4]. classid, score, x, y

        Returns
        -------
        foreground_mask_zero_or_255: np.ndarray
            Predicted foreground mask: [batch, H, W, 3]. 0 or 255

        colored_mask_classid: np.ndarray
            Predicted colored mask: [batch, H, W, 1]. 0 - 24

        keypoints_classidscorexy: np.ndarray
            Predicted keypoints: [batch, N, 4]. classid, score, x, y
        """
        foreground_mask_zero_or_255 = foreground_mask_zero_or_255[0] # 1 batch
        foreground_mask_zero_or_255 = foreground_mask_zero_or_255.transpose(1,2,0).astype(np.uint8) # [H, W, 3]
        colored_mask_classid = colored_mask_classid[0] # 1 batch
        colored_mask_classid = colored_mask_classid.transpose(1,2,0).astype(np.uint8) # [H, W, 1]
        keypoints_classidscorexy = keypoints_classidscorexy[0] # 1 batch
        return foreground_mask_zero_or_255, colored_mask_classid, keypoints_classidscorexy

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

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

def extract_max_score_points_unique(points: np.ndarray, radius: int=20):
    """
    Extracts the points with the maximum score within a specified radius for each classid, ensuring uniqueness.

    Parameters:
    points (np.ndarray): An array of shape [n, 4] where each row represents [classid, score, x, y].
    radius (float): The radius within which to search for points.

    Returns:
    np.ndarray: An array of extracted unique points with the maximum score within the specified radius for each classid.
    """
    # Initialize an empty list to store the maximum score points
    max_score_points = []
    # Iterate over each unique classid in the array
    for classid in np.unique(points[:, 0]):
        class_points = points[points[:, 0] == classid]
        while class_points.size > 0:
            # Take the first point as the reference
            reference_point = class_points[0]
            _, _, ref_x, ref_y = reference_point
            # Calculate the distance from the reference point to all other points in the same class
            distances = np.sqrt((class_points[:, 2] - ref_x)**2 + (class_points[:, 3] - ref_y)**2)
            # Filter points within the specified radius
            within_radius = class_points[distances <= radius]
            # Find the point with the maximum score within the radius
            if within_radius.size > 0:
                max_score_point = within_radius[within_radius[:, 1].argmax()]
                max_score_points.append(max_score_point)
            # Remove the selected points from the class_points
            class_points = np.array([point for point in class_points if list(point) not in within_radius.tolist()])
    # Convert the list of points back to a NumPy array
    return np.array(max_score_points)

def affine_transform(image: np.ndarray, width: int, height: int, dx:int, dy:int):
    # Create a transformation matrix for a parallel shift
    afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
    # apply an infinite transformation
    afin_image = \
        cv2.warpAffine(
            src=image,
            M=afin_matrix,
            dsize=(width, height)
        )
    return afin_image

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-bm',
        '--bodypix_model',
        type=str,
        default='bodypix_resnet50_stride16_1x3x480x640.onnx',
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
        choices=['cpu', 'cuda', 'tensorrt', 'dml', 'coreml'],
        default='tensorrt',
    )
    parser.add_argument(
        '-rt',
        '--runtime',
        type=str,
        choices=['onnx', 'openvino', 'tflite', 'tensorflow'],
        default='onnx',
    )
    parser.add_argument(
        '-s',
        '--strides',
        type=int,
        default=None,
    )
    args = parser.parse_args()

    providers: List[Tuple[str, Dict] | str] = None
    if args.execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'coreml':
        providers = [
            'CoreMLExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'dml':
        providers = [
            'DmlExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.execution_provider == 'tensorrt':
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

    model_bodypix = \
        BodyPix(
            model_path=args.bodypix_model,
            runtime=args.runtime,
            providers=providers,
            strides=args.strides
        )

    cap = cv2.VideoCapture(
        int(args.video) if is_parsable_to_int(args.video) else args.video
    )
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h*2),
    )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)
        mask_image = copy.deepcopy(image)

        debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        # foreground_mask_zero_or_255: [H, W, 3], 0 or 255
        # colored_mask_classid: [H, W, 1], 0 - 24
        # keypoints_classidscorexy: [N, 4] [keypoint_classid, score, x, y]
        foreground_mask_zero_or_255, colored_mask_classid, keypoints_classidscorexy = model_bodypix(debug_image)

        # resize if necessary to match original size
        if foreground_mask_zero_or_255.shape[0] != h or foreground_mask_zero_or_255.shape[1] != w:
            foreground_mask_zero_or_255 = cv2.resize(foreground_mask_zero_or_255, (w, h))

        if colored_mask_classid.shape[0] != h or colored_mask_classid.shape[1] != w:
            colored_mask_classid = cv2.resize(colored_mask_classid, (w, h))

        # Fine-tune position of mask image
        number_of_fine_tuning_pixels: int = model_bodypix.strides // 2
        if number_of_fine_tuning_pixels > 0:
            foreground_mask_zero_or_255 = \
                affine_transform(
                    image=foreground_mask_zero_or_255,
                    height=debug_image_h,
                    width=debug_image_w,
                    dx=-number_of_fine_tuning_pixels,
                    dy=-number_of_fine_tuning_pixels,
                )
            colored_mask_classid = \
                affine_transform(
                    image=colored_mask_classid,
                    height=debug_image_h,
                    width=debug_image_w,
                    dx=-number_of_fine_tuning_pixels,
                    dy=-number_of_fine_tuning_pixels,
                )[..., np.newaxis]

        # Eliminate low score keypoints
        score_keep = keypoints_classidscorexy[..., 1] >= 0.85
        keypoints_classidscorexy = keypoints_classidscorexy[score_keep, :]

        # Eliminate duplicate detection of neighboring keypoints
        if len(keypoints_classidscorexy) > 0:
            keypoints_classidscorexy = extract_max_score_points_unique(keypoints_classidscorexy)

            # scale key-points location to original image
            input_size = np.array([1, 1, *model_bodypix.input_size])
            original_size = np.array([1, 1, debug_image_w, debug_image_h])
            keypoints_classidscorexy[:] = keypoints_classidscorexy[:] / input_size * original_size

        elapsed_time = time.perf_counter() - start_time

        _ = [
            cv2.circle(debug_image, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 0), 2) for landmark in keypoints_classidscorexy
        ]

        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 150, 196), 1, cv2.LINE_AA)

        # binary mask
        mask_image = cv2.bitwise_and(mask_image, foreground_mask_zero_or_255)

        # colored mask
        part_colors = np.asarray(BODY_COLORS, dtype=np.uint8)
        colored_mask = part_colors[colored_mask_classid[..., 0]]
        colored_mask = cv2.bitwise_and(colored_mask, foreground_mask_zero_or_255)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow("test", debug_image)
        cv2.imshow("mask", mask_image)
        cv2.imshow("colored_mask", colored_mask)
        video_writer.write(np.vstack((debug_image, colored_mask)))

    if video_writer:
        video_writer.release()

    if cap:
        cap.release()

if __name__ == "__main__":
    main()
