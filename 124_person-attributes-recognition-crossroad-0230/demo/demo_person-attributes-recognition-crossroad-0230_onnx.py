import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image):
    # ONNX Infomation
    input_name01 = onnx_session.get_inputs()[0].name
    input_name02 = onnx_session.get_inputs()[1].name
    input_name03 = onnx_session.get_inputs()[2].name
    input_name04 = onnx_session.get_inputs()[3].name
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[2]
    input_height = input_size[1]

    # Pre process:Resize, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    result = onnx_session.run(
        None,
        {
            input_name01: input_image,
            input_name02: np.array([0, 0]),
            input_name03: np.array([0, 0]),
            input_name04: np.array([0, 0]),
        },
    )

    # Post process:convert numpy array
    attribute = np.squeeze(result[2])
    top_color_point = np.squeeze(result[1])
    bottom_color_point = np.squeeze(result[0])

    return attribute, top_color_point, bottom_color_point


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float32.onnx',
    )

    args = parser.parse_args()
    model_path = args.model

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # Read image
    image = cv.imread('sample.jpg')

    # Inference
    attribute, top_point, bottom_point = run_inference(onnx_session, image)

    print('is_male        :', attribute[0])
    print('has_bag        :', attribute[1])
    print('has_backpack   :', attribute[2])
    print('has_hat        :', attribute[3])
    print('has_longsleeves:', attribute[4])
    print('has_longpants  :', attribute[5])
    print('has_longhair   :', attribute[6])
    print('has_coat_jacket:', attribute[7])

    image_width, image_height = image.shape[1], image.shape[0]

    top_point_x = int(top_point[0] * image_width)
    top_point_y = int(top_point[1] * image_height)
    cv.circle(image, (top_point_x, top_point_y), 5, (0, 0, 255), -1)

    bottom_point_x = int(bottom_point[0] * image_width)
    bottom_point_y = int(bottom_point[1] * image_height)
    cv.circle(image, (bottom_point_x, bottom_point_y), 5, (255, 0, 0), -1)

    cv.imshow('person-attributes-recognition-crossroad-0230', image)
    cv.waitKey(-1)


if __name__ == '__main__':
    main()
