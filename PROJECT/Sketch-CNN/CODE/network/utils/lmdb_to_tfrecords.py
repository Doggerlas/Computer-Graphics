#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""Convert LMDB to TFRecords
"""

import lmdb
import tensorflow as tf
import os

tfrecord_fn = r'/root/SketchCNN/network/input_tfs/test_input.tfrecords'
data_dir = r'/root/SketchCNN/network/input_pic'


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def lmdb_to_TFRecords():
    writer = tf.io.TFRecordWriter(tfrecord_fn)

    # collect all lmdbs to write into one TFRecords (at least one lmdb)
    #db_paths = [os.path.join(data_dir, 'lmdb_0'), os.path.join(data_dir, 'lmdb_1'), os.path.join(data_dir, 'lmdb_2')]
    db_paths = [os.path.join(data_dir, '')]

    for i in range(1):
        env = lmdb.open(db_paths[i], readonly=True)
        with env.begin() as txn:
            with txn.cursor() as curs:
                for key, value in curs:
                    print('put key: {} to train tfrecord'.format(key.decode('utf-8')))
                    feature = {
                        'name': __bytes_feature(key),
                        'block': __bytes_feature(value)
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    # Set GPU (could remove this setting when running on machine without GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    lmdb_to_TFRecords()
