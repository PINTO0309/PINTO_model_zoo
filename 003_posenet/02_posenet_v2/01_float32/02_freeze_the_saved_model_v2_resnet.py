### tensorflow==2.2.0-rc3

import tensorflow as tf
import os
import shutil
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops

def get_graph_def_from_file(graph_filepath):
  tf.compat.v1.reset_default_graph()
  with ops.Graph().as_default():
    with tf.compat.v1.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.compat.v1.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir,# change input_image to node.name if you know the name
        inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
    )
    print('Optimized graph converted to SavedModel!')



input_name="image"
outputs = ['float_heatmaps:0','float_short_offsets:0','resnet_v1_50/displacement_bwd_2/BiasAdd:0','resnet_v1_50/displacement_fwd_2/BiasAdd:0']

# convert this to a TF Serving compatible mode - posenet_resnet50_16_225
graph_def=get_graph_def_from_file('./posenet_resnet50_16_225.pb')
shutil.rmtree('./saved_model_posenet_resnet50_16_225', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_16_225', './posenet_resnet50_16_225.pb', input_name, outputs)

"""
$ saved_model_cli show --dir saved_model_posenet_resnet50_16_225 --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['image'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 225, 225, 3)
        name: image:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['float_heatmaps'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 15, 15, 17)
        name: float_heatmaps:0
    outputs['float_short_offsets'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 15, 15, 34)
        name: float_short_offsets:0
    outputs['resnet_v1_50/displacement_bwd_2/BiasAdd'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 15, 15, 32)
        name: resnet_v1_50/displacement_bwd_2/BiasAdd:0
    outputs['resnet_v1_50/displacement_fwd_2/BiasAdd'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 15, 15, 32)
        name: resnet_v1_50/displacement_fwd_2/BiasAdd:0
  Method name is: tensorflow/serving/predict
"""

# convert this to a TF Serving compatible mode - saved_model_posenet_resnet50_16_257
graph_def=get_graph_def_from_file('./posenet_resnet50_16_257.pb')
shutil.rmtree('./saved_model_posenet_resnet50_16_257', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_16_257', './posenet_resnet50_16_257.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - saved_model_posenet_resnet50_16_321
graph_def=get_graph_def_from_file('./posenet_resnet50_16_321.pb')
shutil.rmtree('./saved_model_posenet_resnet50_16_321', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_16_321', './posenet_resnet50_16_321.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - saved_model_posenet_resnet50_16_385
graph_def=get_graph_def_from_file('./posenet_resnet50_16_385.pb')
shutil.rmtree('./saved_model_posenet_resnet50_16_385', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_16_385', './posenet_resnet50_16_385.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - saved_model_posenet_resnet50_16_513
graph_def=get_graph_def_from_file('./posenet_resnet50_16_513.pb')
shutil.rmtree('./saved_model_posenet_resnet50_16_513', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_16_513', './posenet_resnet50_16_513.pb', input_name, outputs)



# convert this to a TF Serving compatible mode - posenet_resnet50_32_225
graph_def=get_graph_def_from_file('./posenet_resnet50_32_225.pb')
shutil.rmtree('./saved_model_posenet_resnet50_32_225', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_32_225', './posenet_resnet50_32_225.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - posenet_resnet50_32_257
graph_def=get_graph_def_from_file('./posenet_resnet50_32_257.pb')
shutil.rmtree('./saved_model_posenet_resnet50_32_257', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_32_257', './posenet_resnet50_32_257.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - posenet_resnet50_32_321
graph_def=get_graph_def_from_file('./posenet_resnet50_32_321.pb')
shutil.rmtree('./saved_model_posenet_resnet50_32_321', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_32_321', './posenet_resnet50_32_321.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - posenet_resnet50_32_385
graph_def=get_graph_def_from_file('./posenet_resnet50_32_385.pb')
shutil.rmtree('./saved_model_posenet_resnet50_32_385', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_32_385', './posenet_resnet50_32_385.pb', input_name, outputs)

# convert this to a TF Serving compatible mode - posenet_resnet50_32_513
graph_def=get_graph_def_from_file('./posenet_resnet50_32_513.pb')
shutil.rmtree('./saved_model_posenet_resnet50_32_513', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model_posenet_resnet50_32_513', './posenet_resnet50_32_513.pb', input_name, outputs)