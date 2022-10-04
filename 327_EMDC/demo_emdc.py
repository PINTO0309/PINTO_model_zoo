import onnxruntime as ort
import cv2
import numpy as np


def normalize_rgb(
    img: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
) -> np.ndarray:
    """https://stackoverflow.com/a/71693012/4400024"""
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]

    return img


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "EMDC ONNX inference",
        usage="python main.py [--image-path /path/to/jpg] [--depth-path /path/to/exr] [--width 320] [--height 192]",
    )
    parser.add_argument(
        "--image_path", default="3397.jpg", help="The path to the image file"
    )
    parser.add_argument(
        "--depth_path", default="3397.exr", help="The path to the depth file"
    )

    parser.add_argument("--width", default=320, help="The path to the depth file")
    parser.add_argument("--height", default=256, help="The path to the depth file")

    args = parser.parse_args()

    height = args.height
    width = args.width

    model_path = f"emdc_{ height }x{ width }.onnx"
    ort_sess = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    image = cv2.imread(args.image_path)
    depth = cv2.imread(args.depth_path, cv2.IMREAD_ANYDEPTH)

    image = cv2.resize(image, (width, height))
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

    image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_input = normalize_rgb(image_input)
    image_input = np.transpose(image_input, [2, 0, 1])[np.newaxis, ...]

    sparse_depth = np.zeros_like(depth)
    sparse_depth[0:height:8, 0:width:8] = depth[0:height:8, 0:width:8]

    sparse_depth_input = sparse_depth[np.newaxis, np.newaxis, ...].astype(np.float32)

    outputs = ort_sess.run(
        None, {"image": image_input, "sparse_depth": sparse_depth_input}
    )

    completed_depth = outputs[0][0][0]

    vis_max_depth = 5.0
    cv2.imshow("img", image)
    cv2.imshow("sparse depth", np.clip(sparse_depth, 0, vis_max_depth) / vis_max_depth)
    cv2.imshow("completed depth", np.clip(completed_depth, 0, vis_max_depth) / vis_max_depth)
    cv2.imshow("depth GT", np.clip(depth, 0, vis_max_depth) / vis_max_depth)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
