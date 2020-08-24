### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_pose_detection', source='tensorflow')
mlmodel.save("pose_detection_128x128_float32.mlmodel")

mlmodel = ct.convert('saved_model_pose_landmark_upper_body', source='tensorflow')
mlmodel.save("pose_landmark_upper_body_256x256_float32.mlmodel")
