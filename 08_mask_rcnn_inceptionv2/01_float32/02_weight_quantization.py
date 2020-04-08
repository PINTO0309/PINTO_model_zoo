### tf_nightly-2.2.0.dev20200408

import tensorflow as tf

#tf.compat.v1.enable_eager_execution()

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
#converter.allow_custom_ops = True
tflite_quant_model = converter.convert()
with open('./mask_rcnn_inception_v2_coco_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - mask_rcnn_inception_v2_coco_weight_quant.tflite")

