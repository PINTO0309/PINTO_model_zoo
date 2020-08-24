### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model', source='tensorflow')
mlmodel.save("head_pose_estimator_128x128_float32.mlmodel")
