import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from .backbone import ResNet50Backbone
from .custom_layers import Linear, FixedEmbedding
from .position_embeddings import PositionEmbeddingSine
from .transformer import Transformer
from utils import cxcywh2xyxy


class DETR(tf.keras.Model):
    def __init__(self, num_classes=91, num_queries=100,
                 backbone=None,
                 pos_encoder=None,
                 transformer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.backbone = backbone or ResNet50Backbone(name='backbone')
        self.transformer = transformer or Transformer(return_intermediate_dec=True,
                                                      name='transformer')
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True)

        self.input_proj = Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = FixedEmbedding((num_queries, self.model_dim),
                                          name='query_embed')

        self.class_embed = Linear(num_classes + 1, name='class_embed')

        self.bbox_embed_linear1 = Linear(self.model_dim, name='bbox_embed_0')
        self.bbox_embed_linear2 = Linear(self.model_dim, name='bbox_embed_1')
        self.bbox_embed_linear3 = Linear(4, name='bbox_embed_2')
        self.activation = ReLU()


    def call(self, inp, training=False, post_process=False):
        x, masks = inp
        x = self.backbone(x, training=training)
        masks = self.downsample_masks(masks, x)
        pos_encoding = self.pos_encoder(masks)

        hs = self.transformer(self.input_proj(x), masks, self.query_embed(None),
                              pos_encoding, training=training)[0]

        outputs_class = self.class_embed(hs)

        box_ftmps = self.activation(self.bbox_embed_linear1(hs))
        box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
        outputs_coord = tf.sigmoid(self.bbox_embed_linear3(box_ftmps))

        output = {'pred_logits': outputs_class[-1],
                  'pred_boxes': outputs_coord[-1]}

        if post_process:
            output = self.post_process(output)
        return output


    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [(None, None, None, 3), (None, None, None)]
        super().build(input_shape, **kwargs)


    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        # The existing tf.image.resize with method='nearest'
        # does not expose the half_pixel_centers option in TF 2.2.0
        # The original Pytorch F.interpolate uses it like this
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks


    def post_process(self, output):
        logits, boxes = [output[k] for k in ['pred_logits', 'pred_boxes']]
        
        print('@@@@@@@@@@@@@@ logits', logits)
        print('@@@@@@@@@@@@@@ boxes', boxes)

        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1, name='scores')
        labels = tf.argmax(probs, axis=-1, name='labels')
        boxes = cxcywh2xyxy(boxes)

        # print('@@@@@@@@@@@@ scores', scores.name)
        # print('@@@@@@@@@@@@ labels', labels.name)
        # print('@@@@@@@@@@@@ boxes', boxes.name)

        output = {'scores': scores,
                  'labels': labels,
                  'boxes': boxes}
        return output
