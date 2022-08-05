#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""Network training utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import logging

# util logger initialization
util_logger = logging.getLogger('main.utils')


def slice_tensor(tensor1, tensor2):
    """Slice a new tensor from tensor1 with same H*W shape of tensor2.
    :param tensor1: bigger tensor.
    :param tensor2: smaller tensor.
    :return: sliced tensor.
    """
    with tf.compat.v1.name_scope("slice_tenosr") as _:
        t1_shape = tf.shape(input=tensor1)
        t2_shape = tf.shape(input=tensor2)
		
        offsets = [0, (t1_shape[1] - t2_shape[1]) // 2, (t1_shape[2] - t2_shape[2]) // 2, 0]
        size = [-1, t2_shape[1], t2_shape[2], -1]
        return tf.slice(tensor1, offsets, size)


def make_dir(folder_fn):
    """Create new folder.
    :param folder_fn: folder name.
    :return:
    """
    if tf.io.gfile.exists(folder_fn):
        tf.io.gfile.rmtree(folder_fn)
    tf.io.gfile.makedirs(folder_fn)


def dump_params(path, params):
    """Output all parameters.
    :param path: writen file.
    :param params: parameter dictionary.
    :return:
    """
    util_logger.info('Training settings:')
    with open(path + r'/params.txt', 'w') as f:
        for param in params:
            f.write('{}: {}\n'.format(param, params[param]))
            util_logger.info('{}: {}'.format(param, params[param]))


def cropconcat_layer(tensor1, tensor2, concat_dim=1, name=None):
    """crop tensor1 to have same H,W size with tensor2 and concat them together, used in network building.
    :param tensor1: input tensor bigger one.
    :param tensor2: input smaller one.
    :param concat_dim: concatenate dimension.
    :param name: layer name.
    :return: concatenated tensor.
    """
    with tf.compat.v1.name_scope(name) as _:
        t1_shape = tensor1.get_shape().as_list()
        t2_shape = tensor2.get_shape().as_list()

        if t1_shape[1] != t2_shape[1] and t1_shape[2] != t2_shape[2]:
            offsets = [0, (t1_shape[1] - t2_shape[1]) // 2, (t1_shape[2] - t2_shape[2]) // 2, 0]
            size = [-1, t2_shape[1], t2_shape[2], -1]
            t1_crop = tf.slice(tensor1, offsets, size)
            output = tf.concat([t1_crop, tensor2], concat_dim)
        else:
            output = tf.concat([tensor1, tensor2], concat_dim)

        return output
