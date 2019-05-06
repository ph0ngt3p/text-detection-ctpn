# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

import tensorflow as tf
import numpy as np
# sys.path.append(os.getcwd())
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors


def generate_anchors_tf(height, width, feat_stride=16, anchor_scales=[16, ]):
    shift_x = tf.range(width) * feat_stride  # width
    shift_y = tf.range(height) * feat_stride  # height
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
    K = tf.multiply(width, height)
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

    anchors = generate_anchors(scales=np.array(anchor_scales))
    A = anchors.shape[0]
    anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

    length = K * A
    anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

    return tf.cast(anchors_tf, dtype=tf.float32, name='anchors'), A, length


def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = tf.add(tf.subtract(boxes[:, 2], boxes[:, 0]), 1.0)
    heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    ctr_x = tf.add(boxes[:, 0], widths * 0.5)
    ctr_y = tf.add(boxes[:, 1], heights * 0.5)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im_info):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)


def filter_boxes_tf(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = tf.add(tf.subtract(boxes[:, 2], boxes[:, 0]), 1)
    hs = tf.add(tf.subtract(boxes[:, 3], boxes[:, 1]), 1)
    keep = tf.reshape(tf.where((ws >= min_size) & (hs >= min_size)), [-1])
    return keep


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, feat_stride=[16, ]):
    pre_nms_topN = cfg.RPN_PRE_NMS_TOP_N  # 12000,在做nms之前，最多保留的候选box数目
    post_nms_topN = cfg.RPN_POST_NMS_TOP_N  # 2000，做完nms之后，最多保留的box的数目
    nms_thresh = cfg.RPN_NMS_THRESH  # nms用参数，阈值是0.7
    min_size = cfg.RPN_MIN_SIZE  # 候选box的最小尺寸，目前是16，高宽均要大于16

    im_info = im_info[0]

    height = tf.to_int32(tf.ceil(im_info[0] / tf.cast(feat_stride[0], tf.float32)))
    width = tf.to_int32(tf.ceil(im_info[1] / tf.cast(feat_stride[0], tf.float32)))

    anchors, num_anchors, _ = generate_anchors_tf(height, width)

    # Get the scores and bounding boxes
    scores = tf.reshape(tf.reshape(rpn_cls_prob, [1, height, width, num_anchors, 2])[:, :, :, :, 1],
                        [1, height, width, num_anchors])
    # scores = rpn_cls_prob[:, :, :, num_anchors:]
    scores = tf.reshape(scores, shape=(-1,))
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
    proposals = clip_boxes_tf(proposals, im_info[:2])

    keep = filter_boxes_tf(proposals, min_size)  # 移除那些proposal小于一定尺寸的proposal
    proposals = tf.gather(proposals, keep)
    scores = tf.gather(scores, keep)

    order = tf.nn.top_k(scores, k=pre_nms_topN, sorted=True)[1]  # score按得分的高低进行排序
    proposals = tf.gather(proposals, order)
    scores = tf.gather(scores, order)

    # Non-maximal suppression
    indices = tf.image.non_max_suppression(proposals, scores, post_nms_topN, iou_threshold=nms_thresh)

    boxes = tf.gather(proposals, indices, name='boxes')
    boxes = tf.to_float(boxes)
    scores = tf.gather(scores, indices, name='scores')

    # Only support single image as input
    # batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
    # blob = tf.concat([scores, boxes], 1)

    return boxes, scores


# with tf.get_default_graph().as_default():
#     # im_info = tf.placeholder(tf.float32, shape=[1, 3], name='input_im_info')
#     # cls_prob = tf.placeholder(tf.float32, shape=[1, 54, 380, 2], name='input_im_info')
#     # bbox_pred = tf.placeholder(tf.float32, shape=[1, 54, 38, 40], name='input_im_info')
#     #
#     # blob, scores = proposal_layer_tf(cls_prob, bbox_pred, im_info)
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         height = tf.to_int32(tf.ceil(864 / 16.0))
#         width = tf.to_int32(tf.ceil(608 / 16.0))
#         anchors, num_anchors, _ = generate_anchors_tf(height, width)
#         print(anchors[:, 2])
#         print(sess.run(height))
#         print(sess.run(width))

