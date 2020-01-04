import tensorflow as tf
import os
import shutil
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph

def freeze_model(saved_model_dir, output_node_names, output_filename):
  output_graph_filename = os.path.join(saved_model_dir, output_filename)
  initializer_nodes = ''
  freeze_graph.freeze_graph(
      input_saved_model_dir=saved_model_dir,
      output_graph=output_graph_filename,
      saved_model_tags = tag_constants.SERVING,
      output_node_names=output_node_names,
      initializer_nodes=initializer_nodes,
      input_graph=None,
      input_saver=False,
      input_binary=False,
      input_checkpoint=None,
      restore_op_name=None,
      filename_tensor_name=None,
      clear_devices=True,
      input_meta_graph=False,
  )

def get_graph_def_from_file(graph_filepath):
  tf.reset_default_graph()
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def optimize_graph(model_dir, graph_filename, transforms, input_name, output_names, outname='optimized_model.pb'):
  input_names = [input_name] # change this as per how you have saved the model
  graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,  
      output_names,
      transforms)
  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name=outname)
  print('Graph optimized!')

def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir,# change input_image to node.name if you know the name
        inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
    )
    print('Optimized graph converted to SavedModel!')

tf.compat.v1.enable_eager_execution()

# Look up the name of the placeholder for the input node
graph_def=get_graph_def_from_file('./model-mobilenet_v1_101_225.pb')
input_name=""
for node in graph_def.node:
    if node.op=='Placeholder':
        print("##### model-mobilenet_v1_101_225 - Input Node Name #####", node.name) # this will be the input node
        input_name=node.name

# model-mobilenet_v1_101_225 output names
output_node_names = ['heatmap','offset_2','displacement_fwd_2','displacement_bwd_2']
outputs = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

# Optimizing the graph via TensorFlow library
transforms = []
optimize_graph('./', 'model-mobilenet_v1_101_225.pb', transforms, input_name, output_node_names, outname='optimized_model-mobilenet_v1_101_225.pb')

# convert this to a TF Serving compatible mode - model-mobilenet_v1_101_225
shutil.rmtree('./0', ignore_errors=True)
convert_graph_def_to_saved_model('./0', './optimized_model-mobilenet_v1_101_225.pb', input_name, outputs)



## Look up the name of the placeholder for the input node
#graph_def=get_graph_def_from_file('./model-mobilenet_v1_101_257.pb')
#input_name=""
#for node in graph_def.node:
#    if node.op=='Placeholder':
#        print("##### model-mobilenet_v1_101_257 - Input Node Name #####", node.name) # this will be the input node
#        input_name=node.name

## model-mobilenet_v1_101_257 output names
#output_node_names = ['heatmap','offset_2','displacement_fwd_2','displacement_bwd_2']
#outputs = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

## Optimizing the graph via TensorFlow library
#transforms = []
#optimize_graph('./', 'model-mobilenet_v1_101_257.pb', transforms, input_name, output_node_names, outname='optimized_model-mobilenet_v1_101_257.pb')

## convert this to a TF Serving compatible mode - model-mobilenet_v1_101_257
#shutil.rmtree('./0', ignore_errors=True)
#convert_graph_def_to_saved_model('./0', './optimized_model-mobilenet_v1_101_257.pb', input_name, outputs)



## Look up the name of the placeholder for the input node
#graph_def=get_graph_def_from_file('./model-mobilenet_v1_101_321.pb')
#input_name=""
#for node in graph_def.node:
#    if node.op=='Placeholder':
#        print("##### model-mobilenet_v1_101_321 - Input Node Name #####", node.name) # this will be the input node
#        input_name=node.name

## model-mobilenet_v1_101_321 output names
#output_node_names = ['heatmap','offset_2','displacement_fwd_2','displacement_bwd_2']
#outputs = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

## Optimizing the graph via TensorFlow library
#transforms = []
#optimize_graph('./', 'model-mobilenet_v1_101_321.pb', transforms, input_name, output_node_names, outname='optimized_model-mobilenet_v1_101_321.pb')

## convert this to a TF Serving compatible mode - model-mobilenet_v1_101_321
#shutil.rmtree('./0', ignore_errors=True)
#convert_graph_def_to_saved_model('./0', './optimized_model-mobilenet_v1_101_321.pb', input_name, outputs)



## Look up the name of the placeholder for the input node
#graph_def=get_graph_def_from_file('./model-mobilenet_v1_101_385.pb')
#input_name=""
#for node in graph_def.node:
#    if node.op=='Placeholder':
#        print("##### model-mobilenet_v1_101_385 - Input Node Name #####", node.name) # this will be the input node
#        input_name=node.name

## model-mobilenet_v1_101_385 output names
#output_node_names = ['heatmap','offset_2','displacement_fwd_2','displacement_bwd_2']
#outputs = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

## Optimizing the graph via TensorFlow library
#transforms = []
#optimize_graph('./', 'model-mobilenet_v1_101_385.pb', transforms, input_name, output_node_names, outname='optimized_model-mobilenet_v1_101_385.pb')

## convert this to a TF Serving compatible mode - model-mobilenet_v1_101_385
#shutil.rmtree('./0', ignore_errors=True)
#convert_graph_def_to_saved_model('./0', './optimized_model-mobilenet_v1_101_385.pb', input_name, outputs)



## Look up the name of the placeholder for the input node
#graph_def=get_graph_def_from_file('./model-mobilenet_v1_101_513.pb')
#input_name=""
#for node in graph_def.node:
#    if node.op=='Placeholder':
#        print("##### model-mobilenet_v1_101_513 - Input Node Name #####", node.name) # this will be the input node
#        input_name=node.name

## model-mobilenet_v1_101_513 output names
#output_node_names = ['heatmap','offset_2','displacement_fwd_2','displacement_bwd_2']
#outputs = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

## Optimizing the graph via TensorFlow library
#transforms = []
#optimize_graph('./', 'model-mobilenet_v1_101_513.pb', transforms, input_name, output_node_names, outname='optimized_model-mobilenet_v1_101_513.pb')

## convert this to a TF Serving compatible mode - model-mobilenet_v1_101_513
#shutil.rmtree('./0', ignore_errors=True)
#convert_graph_def_to_saved_model('./0', './optimized_model-mobilenet_v1_101_513.pb', input_name, outputs)
