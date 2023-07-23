# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2023 Katsuya Hyodo

import cv2
import copy
import argparse
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', help='ONNX file path', default='rtmpose_wholebody_m_1x3x256x192_16_with_post.onnx')
    parser.add_argument('--image_file', help='Input image file path', default='human-pose.jpg')
    parser.add_argument('--device', help='device type for inference', default='cpu')
    parser.add_argument('--save_path', help='path to save the output image', default='output.jpg')
    args = parser.parse_args()
    return args


def preprocess(
    img: np.ndarray,
    input_size: Tuple[int, int] = (192, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do preprocessing for RTMPose model inference.

    Args:
        img (np.ndarray): Input image in shape.
        input_size (tuple): Input image size in shape (w, h).

    Returns:
        tuple:
        - resized_img (np.ndarray): Preprocessed image.
        - center (np.ndarray): Center of image.
        - scale (np.ndarray): Scale of image.
    """
    # get shape of image
    img_shape = img.shape[:2]
    # get center and scale
    img_wh = np.asarray([img_shape[1], img_shape[0]], dtype=np.float32)
    center = img_wh * 0.5
    scale = img_wh * 1.25

    # do affine transformation
    resized_img, scale = top_down_affine(input_size, scale, center, img)
    # normalize image
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    resized_img = (resized_img - mean) / std
    return resized_img


def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    """Build onnxruntime session.

    Args:
        onnx_file (str): ONNX file path.
        device (str): Device type for inference.

    Returns:
        sess (ort.InferenceSession): ONNXRuntime session.
    """
    providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, providers=providers)

    return sess


def inference(
    sess: ort.InferenceSession,
    resized_img: np.ndarray,
    img: np.ndarray
) -> np.ndarray:
    """Inference RTMPose model.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        img (np.ndarray): Input image in shape.

    Returns:
        outputs (np.ndarray): Output of RTMPose model.
    """
    # build input
    input = resized_img.transpose(2, 0, 1)[np.newaxis, ...]

    # build output
    sess_input = {
        sess.get_inputs()[0].name: input,
        sess.get_inputs()[1].name: [[img.shape[1], img.shape[0]]],
    }
    sess_output = []
    for out in sess.get_outputs():
        sess_output.append(out.name)

    # run model
    outputs = sess.run(sess_output, sess_input)

    return outputs


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0., 0.),
    inv: bool = False
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift, dtype=np.float32)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5], dtype=np.float32)
    # get four corners of the src rectangle in the original image
    src_dim0 = center + scale * shift
    src_dim1 = center + src_dir + scale * shift
    src = np.concatenate(
        [
            [src_dim0],
            [src_dim1],
            [_get_3rd_point(src_dim0, src_dim1)],
        ],
        axis=0,
        dtype=np.float32,
    )
    # get four corners of the dst rectangle in the input image
    dst_dim0 = np.asarray([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_dim1 = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    dst = np.concatenate(
        [
            [dst_dim0],
            [dst_dim1],
            [_get_3rd_point(dst_dim0, dst_dim1)],
        ],
        axis=0,
        dtype=np.float32,
    )

    if inv:
        warp_mat = cv2.getAffineTransform(dst, src)
    else:
        warp_mat = cv2.getAffineTransform(src, dst)

    return warp_mat.astype(np.float32)


def top_down_affine(
    input_size: dict,
    bbox_scale: dict,
    bbox_center: dict,
    img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_w = bbox_scale[0:1]
    bbox_h = bbox_scale[1:2]
    w_div_075 = bbox_w / 0.75 # 0.75 = model_input_width / model_inut_height
    h_mul_075 = bbox_h * 0.75 # 0.75 = model_input_width / model_inut_height
    w_scaled = np.maximum(h_mul_075, bbox_w)
    h_scaled = np.maximum(w_div_075, bbox_h)
    bbox_scale = np.concatenate([w_scaled, h_scaled], axis=0)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    resized_img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return resized_img, bbox_scale

# default color
skeleton = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (15, 17),
    (15, 18),
    (15, 19),
    (16, 20),
    (16, 21),
    (16, 22),
    (91, 92),
    (92, 93),
    (93, 94),
    (94, 95),
    (91, 96),
    (96, 97),
    (97, 98),
    (98, 99),
    (91, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    (91, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    (91, 108),
    (108, 109),
    (109, 110),
    (110, 111),
    (112, 113),
    (113, 114),
    (114, 115),
    (115, 116),
    (112, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    (112, 121),
    (121, 122),
    (122, 123),
    (123, 124),
    (112, 125),
    (125, 126),
    (126, 127),
    (127, 128),
    (112, 129),
    (129, 130),
    (130, 131),
    (131, 132)
]
palette = [
    [51, 153, 255],
    [0, 255, 0],
    [255, 128, 0],
    [255, 255, 255],
    [255, 153, 255],
    [102, 178, 255],
    [255, 51, 51],
]
link_color = [
    1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
    2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
]
point_color = [
    0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
    4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
]

def visualize(
    image: np.ndarray,
    keypoints: np.ndarray,
    score_threshold=0.3
) -> np.ndarray:
    """Visualize the keypoints and skeleton on image.

    Args:
        img (np.ndarray): Input image in shape.
        keypoints (np.ndarray): Keypoints in image.
        scores (np.ndarray): Model predict scores.
        thr (float): Threshold for visualize.

    Returns:
        img (np.ndarray): Visualized image.
    """
    debug_image = copy.deepcopy(image)
    # draw keypoints and skeleton
    for keypoint in keypoints:
        for (u, v), color in zip(skeleton, link_color):
            score_u = keypoint[u, 2]
            score_v = keypoint[v, 2]
            if score_u > score_threshold and score_v > score_threshold:
                cv2.line(
                    debug_image,
                    tuple(keypoint[u, 0:2].astype(np.int32)),
                    tuple(keypoint[v, 0:2].astype(np.int32)),
                    palette[color],
                    2,
                    cv2.LINE_AA,
                )
    return debug_image

def main():
    args = parse_args()
    # read image from file
    image = cv2.imread(args.image_file)
    sess = build_session(args.onnx_file, args.device)
    h, w = sess.get_inputs()[0].shape[2:]
    model_input_size = (w, h)
    # preprocessing
    resized_img = preprocess(image, model_input_size)
    # inference
    keypoints = inference(sess, resized_img, image)[0]
    # visualize inference result
    debug_image = visualize(image=image, keypoints=keypoints, score_threshold=0.3)
    # save to local
    cv2.imwrite(args.save_path, debug_image)
    cv2.imshow('debug_image', debug_image)
    if cv2.waitKey(0):  # ESC
        pass

if __name__ == '__main__':
    main()
