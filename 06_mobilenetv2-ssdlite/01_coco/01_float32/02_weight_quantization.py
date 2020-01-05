import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./ssdlite_mobilenet_v2_coco_300_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssdlite_mobilenet_v2_coco_300_weight_quant.tflite")

