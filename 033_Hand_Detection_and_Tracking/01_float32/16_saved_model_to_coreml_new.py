### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_hand_landmark_new', source='tensorflow')
mlmodel.save("hand_landmark_new_256x256_float32.mlmodel")
