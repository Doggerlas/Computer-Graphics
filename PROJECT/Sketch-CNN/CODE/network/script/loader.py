#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""TensorFlow multi-thread, queue-based input pipeline.
    The db reader used is TFRecordReader, and we design the custom ops to
    decode raw data into input and label tensors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from libs import decode_block

class SketchReader(object):
    """TFRecords data reader.
            Read data from TFRecord data, based on a multi-thread, queue-based input pipeline.
    """

    def __init__(self, tfrecord_list, raw_size, shuffle=False, num_threads=1, batch_size=1,
                 nb_epoch=None, with_key=False):
        """Reader initializer.

                Args:
                    :param tfrecord_list: tfrecord file lists.
                    :param raw_size: decode raw data size.
                    :param shuffle: shuffle data flag.
                    :param num_threads: number of threads to read data.
                    :param batch_size: batch size.
                    :param nb_epoch: number of epochs, 1 if final test phase
                    :param with_key: the key from extracted key-value pair.
        """
        self._reader = None
        self._queue = None
        self._tfrecord_list = tfrecord_list
        self._raw_size = raw_size
        self._shuffle = shuffle
        self._nb_threads = num_threads
        self._batch_size = batch_size
        self._nb_epoch = nb_epoch
        self._with_key = with_key

    def _read_raw(self):
        """Read raw data from TFRecord.

        Returns:
            :return: data list [input_raw, label_raw].
        """
        # 生成TFRecord Reader
        self._reader = tf.compat.v1.TFRecordReader()
        # 读取tfrecord文件，读取得到的是一个序列化的example
        _, serialized_example = self._reader.read(self._queue)
        # print("序列化的example：serialized_example：", serialized_example)# Tensor("ReaderReadV2:1", shape=(), dtype=string)
        # 解析得到的系列化example，需要按照存储时的格式还原features，必须写明features内的字典的键索引得到特定的数据！
        features = tf.io.parse_single_example(serialized=serialized_example,
                                           features={
                                               'name': tf.io.FixedLenFeature([], tf.string),
                                               'block': tf.io.FixedLenFeature([], tf.string)
                                           })

        input_raw, label_raw = decode_block(features['block'], tensor_size=self._raw_size)#decode block输入进去的是features['block'],这个是一个字符串，是对训练集中Sketch和3D图形图片的压缩 ；输出的是两个代表inputdata和label的tensor
        # decode_block的输入
        # print("features['name']打印值", features['name'])# Tensor("ParseSingleExample/ParseExample/ParseExampleV2:1", shape=(), dtype=string)
        # print("features['block']打印值", features['block'])#features['block']打印值 Tensor("ParseSingleExample/ParseExample/ParseExampleV2:0", shape=(), dtype=string)
        if self._with_key:
            return input_raw, label_raw, features['name']
        # print("input_raw打印值", input_raw)#Tensor("DecodeBlock:0", shape=(256, 256, 6), dtype=float32)
        # print("label_raw打印值", label_raw)#Tensor("DecodeBlock:1", shape=(256, 256, 17), dtype=float32)
        return input_raw, label_raw

    def _batch_data(self):
        """Assemble data into one batch.

        Returns:
            :return: batch data with shape [N, H, W, C].
        """
        # 生成文件队列
        self._queue = tf.compat.v1.train.string_input_producer(self._tfrecord_list,
                                                     num_epochs=self._nb_epoch,
                                                     shuffle=self._shuffle)

        example = self._read_raw()

        queue_buf = 500
        cap_shuffle = queue_buf + 3 * self._batch_size
        cap_noShuffle = (self._nb_threads + 1) * self._batch_size

        if self._shuffle:
            batch_data = tf.compat.v1.train.shuffle_batch(
                tensors=example,
                batch_size=self._batch_size,
                num_threads=self._nb_threads,
                capacity=cap_shuffle,
                min_after_dequeue=queue_buf
            )
        else:
            batch_data = tf.compat.v1.train.batch(
                tensors=example,
                batch_size=self._batch_size,
                num_threads=self._nb_threads,
                capacity=cap_noShuffle
            )
        return batch_data

    def next_batch(self):
        """Load next batch

        Returns:
            :return: next batch data.
        """
        return self._batch_data()
