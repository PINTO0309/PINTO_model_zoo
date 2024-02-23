import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
np.random.seed(0)
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-b',
        '--batches',
        type=int,
        default=1,
        help='batch size'
    )
    parser.add_argument(
        '-x',
        '--boxes',
        type=int,
        default=5040,
        help='boxes'
    )
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        default=80,
        help='classes'
    )
    args = parser.parse_args()
    BATCHES = args.batches
    BOXES = args.boxes
    CLASSES = args.classes


    # Create a model
    scores = tf.keras.layers.Input(
        shape=[
            CLASSES,
            BOXES,
        ],
        batch_size=BATCHES,
        dtype=tf.float32,
    )

    selected_indices = tf.keras.layers.Input(
        shape=[
            3,
        ],
        dtype=tf.int64,
    )

    gathered_scores = tf.gather_nd(
        scores,
        selected_indices,
        batch_dims=0,
    )
    expands_scores = gathered_scores[:,np.newaxis]

    model = tf.keras.models.Model(inputs=[scores,selected_indices], outputs=[expands_scores])
    model.summary()
    output_path = 'saved_model_postprocess'
    tf.saved_model.save(model, output_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    open(f"{output_path}/nms_score_gather_nd.tflite", "wb").write(tflite_model)