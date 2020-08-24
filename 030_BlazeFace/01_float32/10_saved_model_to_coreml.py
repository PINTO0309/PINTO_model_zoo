### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_face_detection_front', source='tensorflow')
mlmodel.save("face_detection_front_128x128_float32.mlmodel")

mlmodel = ct.convert('saved_model_face_detection_back', source='tensorflow')
mlmodel.save("face_detection_back_256x256_float32.mlmodel")
