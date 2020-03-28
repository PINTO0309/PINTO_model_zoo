import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm
import sys
from tensorflow.python.saved_model import tag_constants
from tensorflow.python import ops
import shutil
import json
import time

def animation(load_folder, save_folder, model_path):
    #input_photo = tf.placeholder(tf.float32, [1, 360, 360, 3], name='input')
    #network_out = network.unet_generator(input_photo)
    #final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
    #print("input_photo.name =", input_photo.name)
    #print("input_photo.shape =", input_photo.shape)
    #print("final_out.name =", final_out.name)
    #print("final_out.shape =", final_out.shape)
    #all_vars = tf.trainable_variables()
    #gene_vars = [var for var in all_vars if 'generator' in var.name]
    #saver = tf.train.Saver(var_list=gene_vars)
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())
    #saver.restore(sess, tf.train.latest_checkpoint(model_path))
    #saver.save(sess, './export/model_360.ckpt')
    #sys.exit(0)

    #graph = tf.get_default_graph()
    #sess = tf.Session()
    #saver = tf.train.import_meta_graph('./export/model_360.ckpt.meta')
    #saver.restore(sess, './export/model_360.ckpt')
    #tf.train.write_graph(sess.graph_def, './export', 'white_box_cartoonization_freeze_graph_360.pbtxt', as_text=True)
    #tf.train.write_graph(sess.graph_def, './export', 'white_box_cartoonization_freeze_graph_360.pb', as_text=False)
    #sys.exit(0)


    #def get_graph_def_from_file(graph_filepath):
    #    tf.compat.v1.reset_default_graph()
    #    with ops.Graph().as_default():
    #        with tf.compat.v1.gfile.GFile(graph_filepath, 'rb') as f:
    #            graph_def = tf.compat.v1.GraphDef()
    #            graph_def.ParseFromString(f.read())
    #            return graph_def

    #def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
    #    graph_def = get_graph_def_from_file(graph_filepath)
    #    with tf.compat.v1.Session(graph=tf.Graph()) as session:
    #        tf.import_graph_def(graph_def, name='')
    #        tf.compat.v1.saved_model.simple_save(
    #            session,
    #            export_dir,# change input_image to node.name if you know the name
    #            inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
    #                for node in graph_def.node if node.op=='Placeholder'},
    #            outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
    #        )
    #        print('Optimized graph converted to SavedModel!')
    
    #shutil.rmtree('./saved_model', ignore_errors=True)
    #convert_graph_def_to_saved_model('./saved_model', './export/white_box_cartoonization_freeze_graph_360.pb', 'input', ['add_1:0'])
    #sys.exit(0)

    with tf.Session() as sess:
        with tf.gfile.GFile('./export/white_box_cartoonization_freeze_graph_720.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            _ = tf.import_graph_def(graph_def)

            tensor_input = sess.graph.get_tensor_by_name('import/input:0')
            tensor_output = sess.graph.get_tensor_by_name('import/add_1:0')
            
            #ops = {}
            #for op in tf.get_default_graph().get_operations():
            #    ops[op.name] = [str(output) for output in op.outputs]
            #with open('./export/white_box_cartoonization_freeze_graph_360.json', 'w') as f:
            #    f.write(json.dumps(ops))
            #sys.exit(0)

            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FPS, 30)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            window_name = "USB Camera"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            framecount = 0
            fps = ""
            time1 = 0

            while True:
                start_time = time.perf_counter()

                ret, image = cam.read()
                if not ret:
                    continue

                size = 720
                colw = image.shape[1]
                colh = image.shape[0]
                new_w = int(colw * min(size/colw, size/colh))
                new_h = int(colh * min(size/colw, size/colh))
                resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_AREA)
                canvas = np.full((size, size, 3), 0)
                canvas[(size - new_h)//2:(size - new_h)//2 + new_h,(size - new_w)//2:(size - new_w)//2 + new_w, :] = resized_image
                image = canvas

                batch_image = image.astype(np.float32)/127.5 - 1
                batch_image = np.expand_dims(batch_image, axis=0)
                output = sess.run(tensor_output, {tensor_input: batch_image})

                output = (np.squeeze(output)+1)*127.5
                output = np.clip(output, 0, 255).astype(np.uint8)

                cv2.putText(output, "---- Animation ---- " + fps, (640 - 550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (38, 0, 255), 1, cv2.LINE_AA)
                image = image.astype(np.uint8)
                cv2.putText(image, "---- Ground truth ---- " + fps, (640 - 550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (38, 0, 255), 1, cv2.LINE_AA)
                img =  np.hstack((image, output))
                cv2.imshow('USB Camera', img)

                if cv2.waitKey(1)&0xFF == ord('q'):
                    break
    

                # FPS calculation
                framecount += 1
                if framecount >= 10:
                    fps = "(Playback) {:.1f} FPS".format(time1 / 10)
                    framecount = 0
                    time1 = 0
                end_time = time.perf_counter()
                elapsedTime = end_time - start_time
                time1 += 1 / elapsedTime

if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    animation(load_folder, save_folder, model_path)
    

    