### tensorflow==2.3.0

import tensorflow as tf
import coremltools as ct

#mlmodel = ct.convert('saved_model_deeplab_v3_plus_mnv2_aspp_decoder_256', source='tensorflow')
#mlmodel.save("deeplab_v3_plus_mnv2_aspp_decoder_256_float32.mlmodel")

mlmodel = ct.convert('saved_model_deeplab_v3_plus_mnv2_decoder_256', source='tensorflow')
mlmodel.save("deeplab_v3_plus_mnv2_decoder_256_float32.mlmodel")

#mlmodel = ct.convert('saved_model_deeplab_v3_plus_mnv2_decoder_513', source='tensorflow')
#mlmodel.save("deeplab_v3_plus_mnv2_decoder_513_float32.mlmodel")

#mlmodel = ct.convert('saved_model_deeplab_v3_plus_mnv3_decoder_256', source='tensorflow')
#mlmodel.save("deeplab_v3_plus_mnv3_decoder_256_float32.mlmodel")