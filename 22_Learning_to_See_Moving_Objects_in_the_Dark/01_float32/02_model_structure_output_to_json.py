
import tensorflow as tf 
import json

with tf.Session() as sess:
    with tf.gfile.GFile('./1_checkpoint/16_bit_HE_to_HE_gt/lsmod_256.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        _ = tf.import_graph_def(graph_def)
        ops = {}
        for op in tf.get_default_graph().get_operations():
            ops[op.name] = [str(output) for output in op.outputs]
        with open('./1_checkpoint/16_bit_HE_to_HE_gt/lsmod_256.json', 'w') as f:
            f.write(json.dumps(ops))