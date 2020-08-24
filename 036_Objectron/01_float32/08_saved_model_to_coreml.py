### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_object_detection_3d_chair', source='tensorflow')
mlmodel.save("object_detection_3d_chair_640x480_float32.mlmodel")

mlmodel = ct.convert('saved_model_object_detection_3d_sneakers', source='tensorflow')
mlmodel.save("object_detection_3d_sneakers_640x480_float32.mlmodel")