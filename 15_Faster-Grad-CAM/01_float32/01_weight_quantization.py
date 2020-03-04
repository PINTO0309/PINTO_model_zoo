import tensorflow as tf

# Weight Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 96 x 96 x 3)
# OUTPUT = block_16_expand_relu, global_average_pooling2d_1
model = tf.keras.models.model_from_json(open('model_for_quantization.json').read())
model.load_weights('weights.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./weights_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - weights_weight_quant.tflite")

