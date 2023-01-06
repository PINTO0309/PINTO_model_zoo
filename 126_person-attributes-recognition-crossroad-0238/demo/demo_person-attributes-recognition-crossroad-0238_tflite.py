import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, image):
    # TFLite Infomation
    input_details = interpreter.get_input_details()
    input_size = input_details[0]['shape']
    input_width = input_size[2]
    input_height = input_size[1]

    # Pre process:Resize, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Post process:convert numpy array
    output_details = interpreter.get_output_details()
    attribute = interpreter.get_tensor(output_details[0]['index'])
    attribute = np.squeeze(attribute)

    return attribute


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float32.tflite',
    )

    args = parser.parse_args()
    model_path = args.model

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Read image
    image = cv.imread('sample.jpg')

    # Inference
    attribute = run_inference(interpreter, image)

    print('is_male        :', attribute[0])
    print('has_bag        :', attribute[1])
    print('has_hat        :', attribute[2])
    print('has_longsleeves:', attribute[3])
    print('has_longpants  :', attribute[4])
    print('has_longhair   :', attribute[5])
    print('has_coat_jacket:', attribute[6])

    cv.imshow('person-attributes-recognition-crossroad-0238', image)
    cv.waitKey(-1)


if __name__ == '__main__':
    main()
