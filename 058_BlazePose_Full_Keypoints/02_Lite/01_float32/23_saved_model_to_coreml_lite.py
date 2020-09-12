### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

mlmodel = ct.convert('saved_model_lite_pose_detection', source='tensorflow')
mlmodel.save("lite_pose_detection_full_128x128_float32.mlmodel")

mlmodel = ct.convert('saved_model_lite_pose_landmark_39kp', source='tensorflow')
mlmodel.save("lite_pose_landmark_full_body_256x256_float32.mlmodel")
