### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_hand_landmark', source='tensorflow')
mlmodel.save("hand_landmark_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_hand_landmark_3d', source='tensorflow')
mlmodel.save("hand_landmark_3d_256x256_float32.mlmodel")

mlmodel = ct.convert('saved_model_palm_detection_builtin', source='tensorflow')
mlmodel.save("palm_detection_builtin_256x256_float32.mlmodel")