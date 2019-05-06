from __future__ import print_function

import os
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.rpn_msr.proposal_layer_tf import proposal_layer_tf
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('export_path', 'exported/1', '')
FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
    with tf.get_default_graph().as_default():
        image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(image)
        boxes, scores = proposal_layer_tf(cls_prob, bbox_pred, im_info)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.export_path)
            freezing_graph = sess.graph

            input_img = freezing_graph.get_tensor_by_name('input_image:0')
            input_im_info = freezing_graph.get_tensor_by_name('input_im_info:0')
            output_boxes = freezing_graph.get_tensor_by_name('boxes:0')
            output_scores = freezing_graph.get_tensor_by_name('scores:0')

            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'image': tf.saved_model.utils.build_tensor_info(input_img),
                    'im_info': tf.saved_model.utils.build_tensor_info(input_im_info),
                },
                outputs={
                    'boxes': tf.saved_model.utils.build_tensor_info(output_boxes),
                    'scores': tf.saved_model.utils.build_tensor_info(output_scores),
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        signature
                },
                legacy_init_op=legacy_init_op,
                clear_devices=True
            )

            builder.save()
