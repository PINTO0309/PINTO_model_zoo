import tensorflow as tf
import numpy as np

def representative_dataset_gen():
    raw_test_data = np.load('janken_dataset.npy')
    for image in raw_test_data:
        image = tf.image.resize(image, (96, 96))
        image = image / 255
        calibration_data = image[np.newaxis, :, :, :]
        yield [calibration_data]

# Integer Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 96 x 96 x 3)
# OUTPUT = block_16_expand_relu, global_average_pooling2d_1
model = tf.keras.models.model_from_json(open('model_for_quantization.json').read())
model.load_weights('weights.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./weights_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - weights_integer_quant.tflite")
