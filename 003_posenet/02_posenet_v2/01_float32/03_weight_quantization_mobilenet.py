### tensorflow==2.2.0-rc3

import tensorflow as tf
import numpy as np

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_8_225')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_8_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_8_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_8_257')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_8_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_8_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_8_321')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_8_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_8_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_8_385')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_8_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_8_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_8_513')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_8_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_8_513_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_16_225')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_16_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_16_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_16_257')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_16_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_16_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_16_321')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_16_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_16_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_16_385')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_16_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_16_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm050_16_513')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm050_16_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm050_16_513_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_8_225')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_8_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_8_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_8_257')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_8_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_8_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_8_321')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_8_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_8_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_8_385')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_8_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_8_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_8_513')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_8_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_8_513_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_16_225')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_16_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_16_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_16_257')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_16_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_16_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_16_321')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_16_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_16_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_16_385')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_16_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_16_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm075_16_513')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm075_16_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm075_16_513_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_8_225')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_8_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_8_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_8_257')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_8_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_8_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_8_321')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_8_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_8_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_8_385')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_8_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_8_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_8_513')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_8_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_8_513_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_16_225')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_16_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_16_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_16_257')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_16_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_16_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_16_321')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_16_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_16_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_16_385')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_16_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_16_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenetv1_dm100_16_513')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenetv1_dm100_16_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenetv1_dm100_16_513_weight_quant.tflite")