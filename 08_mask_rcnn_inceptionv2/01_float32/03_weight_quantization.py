import tensorflow as tf

### Tensorflow v2.1.0 - master - commit 8fdb834931fe62abaeab39fe6bf0bcc9499b25bf

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open('./mask_rcnn_inception_v2_coco_800_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - mask_rcnn_inception_v2_coco_800_weight_quant.tflite")

