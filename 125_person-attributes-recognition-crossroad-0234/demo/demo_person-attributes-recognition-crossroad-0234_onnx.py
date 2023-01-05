import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image):
    # ONNX Infomation
    input_name01 = onnx_session.get_inputs()[0].name
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[2]
    input_height = input_size[1]

    # Pre process:Resize, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    result = onnx_session.run(None, {input_name01: input_image})

    # Post process:convert numpy array
    attribute = np.squeeze(result[0])

    return attribute


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
    attribute = run_inference(onnx_session, image)

    print('is_male        :', attribute[0])
    print('has_bag        :', attribute[1])
    print('has_hat        :', attribute[2])
    print('has_longsleeves:', attribute[3])
    print('has_longpants  :', attribute[4])
    print('has_longhair   :', attribute[5])
    print('has_coat_jacket:', attribute[6])

    cv.imshow('person-attributes-recognition-crossroad-0234', image)
    cv.waitKey(-1)


if __name__ == '__main__':
    main()
