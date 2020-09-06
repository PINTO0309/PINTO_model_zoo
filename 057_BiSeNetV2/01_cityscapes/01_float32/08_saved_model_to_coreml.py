### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_256x256', source='tensorflow')
mlmodel.save("bisenetv2_cityscapes_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_480x640', source='tensorflow')
mlmodel.save("bisenetv2_cityscapes_480x640_float32.mlmodel")

mlmodel = ct.convert('saved_model_512x1024', source='tensorflow')
mlmodel.save("bisenetv2_cityscapes_512x1024_float32.mlmodel")

