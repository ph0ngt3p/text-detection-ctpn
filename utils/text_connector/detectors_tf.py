import tensorflow as tf
from .text_connect_cfg import Config as TextLineCfg


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes0 = tf.maximum(tf.minimum(boxes[:, 0], im_shape[1] - 1), 0)
    boxes1 = tf.maximum(tf.minimum(boxes[:, 1], im_shape[0] - 1), 0)
    return tf.stack([boxes0, boxes1], axis=1)


def filter_boxes(boxes):
    heights = tf.map_fn(lambda b: (tf.abs(b[5] - b[1]) + tf.abs(b[7] - b[3]) / 2.0 + 1), boxes)
    widths = tf.map_fn(lambda b: (tf.abs(b[2] - b[0]) + tf.abs(b[6] - b[4]) / 2.0 + 1), boxes)
    scores = tf.map_fn(lambda b: b[8], boxes)
    return tf.reshape(tf.where((widths / heights > TextLineCfg.MIN_RATIO) & (scores > TextLineCfg.LINE_MIN_SCORE) &
                               (widths > (TextLineCfg.TEXT_PROPOSALS_WIDTH * TextLineCfg.MIN_NUM_PROPOSALS))), [-1])


def detect(text_proposals, scores, size):
    # 删除得分较低的proposal
    keep = tf.reshape(tf.where(scores > TextLineCfg.TEXT_PROPOSALS_MIN_SCORE), [-1])
    text_proposals = tf.gather(text_proposals, keep)
    scores = tf.gather(scores, keep)

    # 按得分排序
    sorted_indices = tf.argsort(tf.reshape(scores, [-1]))[::-1]
    text_proposals = tf.gather(text_proposals, sorted_indices)
    scores = tf.gather(scores, sorted_indices)

    # 对proposal做nms
    nms_indices = tf.image.non_max_suppression(text_proposals,
                                               scores,
                                               tf.shape(scores)[0],
                                               TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
    text_proposals = tf.gather(text_proposals, nms_indices)
    scores = tf.gather(scores, nms_indices)

    scores = scores[:, tf.newaxis]

    text_proposals = tf.Print(text_proposals, [text_proposals, scores], 'tf text prop, tf scores')

    # 获取检测结果
    text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
    keep_inds = self.filter_boxes(text_recs)
    return text_recs[keep_inds]