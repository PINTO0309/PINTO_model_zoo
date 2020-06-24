

import tensorflow as tf
def init(*args):

    if len(args)==1:
        use_pb=True
        pb_path=args[0]
    else:
        use_pb=False
        meta_path=args[0]
        restore_model_path=args[1]

    def ini_ckpt():
        graph = tf.Graph()
        graph.as_default()
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        sess = tf.Session(config=configProto)
        #load_model(model_path, sess)
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, restore_model_path)

        print("Model restred!")
        return (graph, sess)
    def init_pb(model_path):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        compute_graph = tf.Graph()
        compute_graph.as_default()
        sess = tf.Session(config=config)
        with tf.gfile.GFile(model_path, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')

        # saver = tf.train.Saver(tf.global_variables())
        # saver.save(sess, save_path='./tmp.ckpt')
        return (compute_graph, sess)

    if use_pb:
        model = init_pb(pb_path)
    else:
        model = ini_ckpt()

    graph = model[0]
    sess = model[1]

    return graph,sess