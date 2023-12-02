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
    args = parser.parse_args()
    BATCHES = args.batches
    BOXES = args.boxes

    # Create a model
    boxes = tf.keras.layers.Input(
        shape=[
            BOXES,
            4,
        ],
        batch_size=BATCHES,
        dtype=tf.float32,
    )

    selected_indices = tf.keras.layers.Input(
        shape=[
            2,
        ],
        dtype=tf.int64,
    )

    gathered_boxes = tf.gather_nd(
        boxes,
        selected_indices,
        batch_dims=0,
    )
    gathered_boxes_casted = tf.cast(gathered_boxes, dtype=tf.float32)

    model = tf.keras.models.Model(inputs=[boxes, selected_indices], outputs=[gathered_boxes_casted])
    model.summary()
    output_path = 'saved_model_postprocess'
    tf.saved_model.save(model, output_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    open(f"{output_path}/nms_box_gather_nd.tflite", "wb").write(tflite_model)