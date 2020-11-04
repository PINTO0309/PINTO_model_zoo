import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
import sys

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow_datasets as tfds

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True, help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet", help="model architecture ('srresnet' or 'unet')")
    # parser.add_argument("--weight_file", type=str, required=True, help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25", help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None, help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    height = 512
    width  = 512

    noise  = 'gauss'
    mode   = 'clean'

    args = get_args()
    image_dir = args.image_dir
    weight_file = 'weights_{}_{}.hdf5'.format(noise, mode) #args.weight_file

    if mode != 'clean':
        val_noise_model = get_noise_model(args.test_noise_model)
    else:
        model = get_model(height, width, args.model)
    model.load_weights(weight_file)
    model.summary()

    # saved_model
    tf.saved_model.save(model, 'saved_model_{}_{}_{}_{}x{}'.format(args.model, noise, mode, height, width))

    # pb
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)])
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=".",
                        name="noise2noise_{}_{}_{}_{}x{}_float32.pb".format(args.model, noise, mode, height, width),
                        as_text=False)

    # No Quantization - Input/Output=float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('noise2noise_{}_{}_{}_{}x{}_float32.tflite'.format(args.model, noise, mode, height, width), 'wb') as w:
        w.write(tflite_model)
    print("tflite convert complete! - noise2noise_{}_{}_{}_{}x{}_float32.tflite".format(args.model, noise, mode, height, width))


    # Weight Quantization - Input/Output=float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    with open('noise2noise_{}_{}_{}_{}x{}_weight_quant.tflite'.format(args.model, noise, mode, height, width), 'wb') as w:
        w.write(tflite_model)
    print('Weight Quantization complete! - noise2noise_{}_{}_{}_{}x{}_weight_quant.tflite'.format(args.model, noise, mode, height, width))

    # Float16 Quantization - Input/Output=float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    with open('noise2noise_{}_{}_{}_{}x{}_float16_quant.tflite'.format(args.model, noise, mode, height, width), 'wb') as w:
        w.write(tflite_quant_model)
    print('Float16 Quantization complete! - noise2noise_{}_{}_{}_{}x{}_float16_quant.tflite'.format(args.model, noise, mode, height, width))


    def representative_dataset_gen():
        for data in raw_test_data.take(10):
            image = data['image'].numpy()
            image = tf.image.resize(image, (height, width))
            image = image[np.newaxis,:,:,:]
            # image = image / 127.5 - 1.0
            yield [image]

    raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)


    # Integer Quantization - Input/Output=float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    with open('noise2noise_{}_{}_{}_{}x{}_integer_quant.tflite'.format(args.model, noise, mode, height, width), 'wb') as w:
        w.write(tflite_quant_model)
    print('Integer Quantization complete! - noise2noise_{}_{}_{}_{}x{}_integer_quant.tflite'.format(args.model, noise, mode, height, width))


    # Full Integer Quantization - Input/Output=int8
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    with open('noise2noise_{}_{}_{}_{}x{}_full_integer_quant.tflite'.format(args.model, noise, mode, height, width), 'wb') as w:
        w.write(tflite_quant_model)
    print('Integer Quantization complete! - noise2noise_{}_{}_{}_{}x{}_full_integer_quant.tflite'.format(args.model, noise, mode, height, width))


    # # EdgeTPU
    # import subprocess
    # result = subprocess.check_output(["edgetpu_compiler", "-s", "noise2noise_{}_{}_{}_{}x{}_full_integer_quant.tflite".format(args.model, noise, mode, height, width)])
    # print(result)



    sys.exit(0)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        h, w, _ = image.shape

        out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        noise_image = val_noise_model(image)
        pred = model.predict(np.expand_dims(noise_image, 0))
        denoised_image = get_image(pred[0])
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
