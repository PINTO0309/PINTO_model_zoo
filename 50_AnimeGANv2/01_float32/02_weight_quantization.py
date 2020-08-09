### tensorflow==2.3.0

import tensorflow as tf

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_Hayao')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('animeganv2_hayao_256x256_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - animeganv2_hayao_256x256_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_Paprika')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('animeganv2_paprika_256x256_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - animeganv2_paprika_256x256_weight_quant.tflite")

