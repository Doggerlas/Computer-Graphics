#
# 采用原始decode方式进行数据输入
#
# python3 test_geomNet_org.py --cktDir=../output/train_geomNet/savedModel --dbTest=../sampleData/test --outDir=../output/test/test_geom_sampleData_Net --device=0,1  --graphName=SAS_2stage_sampleData_GeoNet.pbtxt
#
# ==============================================================================
"""Geometry regression network testing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import argparse

import tensorflow.compat.v1 as tf
from network import SKETCHNET
from loader import SketchReader
from utils.util_func import slice_tensor, make_dir, dump_params

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

# Hyper Parameters
hyper_params = {
    'dbTest': '',
    'outDir': '',
    'device': '0',
    'rootFt': 32,
    'cktDir': '',
    'nbThreads': 1,
    'dsWeight': 5.0,
    'regWeight': 2.0,
    'dlossScale': 920.0,
    'nlossScale': 430.0,
    'graphName': '',
}

tf.disable_eager_execution()

# test_real_input切分成13个(1,256,256,1)的向量
nprLine_input = tf.placeholder(tf.float32, [None, None, None, 1], name='npr_input')
ds_input = tf.placeholder(tf.float32, [None, None, None, 1], name='ds_input')
fm_input = tf.placeholder(tf.float32, [None, None, None, 1], name='fLMask_input')
fmInv_input = tf.placeholder(tf.float32, [None, None, None, 1], name='fLInvMask_input')
gtNormal_input = tf.placeholder(tf.float32, [None, None, None, 3], name='gtN_input')
gtDepth_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtD_input')
gtField_input = tf.placeholder(tf.float32, [None, None, None, 4], name='gtField_input')
clineInvMask_input = tf.placeholder(tf.float32, [None, None, None, 1], name='clIMask_input')
maskShape_input = tf.placeholder(tf.float32, [None, None, None, 1], name='shapeMask_input')
maskDs_input = tf.placeholder(tf.float32, [None, None, None, 1], name='dsMask_input')
mask2D_input = tf.placeholder(tf.float32, [None, None, None, 1], name='2dMask_input')
selLineMask_input = tf.placeholder(tf.float32, [None, None, None, 1], name='sLMask_input')
vdotnScalar_input = tf.placeholder(tf.float32, [None, None, None, 1], name='curvMag_input')


# depth, normal regularization term
def reg_loss(logit_n, logit_d, shape_mask, cl_mask_inverse, fl_mask_inv, scope='reg_loss'):
    with tf.name_scope(scope) as _:
        # convert normal signal back to [-1, 1]
        converted_n = (logit_n * 2.0) - 1.0

        img_shape = tf.shape(logit_d)
        N = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        K = 0.007843137254902

        shape_mask_crop = slice_tensor(shape_mask, logit_d)
        l_mask_crop = slice_tensor(cl_mask_inverse, logit_d)
        fl_mask_inv_crop = slice_tensor(fl_mask_inv, logit_d)
        combined_mask = shape_mask_crop * l_mask_crop * fl_mask_inv_crop
        mask_shift_x = tf.slice(combined_mask, [0, 0, 0, 0], [-1, -1, W - 1, -1])
        mask_shift_y = tf.slice(combined_mask, [0, 0, 0, 0], [-1, H - 1, -1, -1])

        c0 = tf.fill([N, H, W - 1, 1], K)
        c1 = tf.zeros(shape=[N, H, W - 1, 1])

        cx = logit_d[:, :, 1:, :] - logit_d[:, :, :-1, :]
        t_x = tf.concat([c0, c1, cx], axis=3)
        # approximate normalization
        t_x /= K

        c2 = tf.zeros(shape=[N, H - 1, W, 1])
        c3 = tf.fill([N, H - 1, W, 1], K)
        cy = logit_d[:, 1:, :, :] - logit_d[:, :-1, :, :]
        t_y = tf.concat([c2, c3, cy], axis=3)
        # approximate normalization
        t_y /= K

        normal_shift_x = tf.slice(converted_n, [0, 0, 0, 0], [-1, -1, W - 1, -1])
        normal_shift_y = tf.slice(converted_n, [0, 0, 0, 0], [-1, H - 1, -1, -1])

        reg_loss1_diff = tf.reduce_sum(t_x * normal_shift_x, 3)
        reg_loss1 = tf.losses.mean_squared_error(tf.zeros(shape=[N, H, W - 1]), reg_loss1_diff,
                                                 weights=tf.squeeze(mask_shift_x, [3]))

        reg_loss2_diff = tf.reduce_sum(t_y * normal_shift_y, 3)
        reg_loss2 = tf.losses.mean_squared_error(tf.zeros(shape=[N, H - 1, W]), reg_loss2_diff,
                                                 weights=tf.squeeze(mask_shift_y, [3]))

        return reg_loss1 + reg_loss2


# total loss
def loss(logit_d, logit_n, logit_c, normal, depth, shape_mask, ds_mask, cl_mask_inverse,
         gt_ds, npr, logit_f, gt_f, fl_mask_inv):
    img_shape = tf.shape(logit_d)
    N = img_shape[0]
    H = img_shape[1]
    W = img_shape[2]

    mask_crop = slice_tensor(shape_mask, logit_n)
    mask_crop3 = tf.tile(mask_crop, [1, 1, 1, 3])

    zero_tensor = tf.zeros(shape=[N, H, W, 1])
    zero_tensor3 = tf.zeros(shape=[N, H, W, 3])
    logit_c3 = tf.tile(logit_c, [1, 1, 1, 3])

    # normal loss (l2)
    gt_normal = slice_tensor(normal, logit_n)
    n_loss = tf.losses.mean_squared_error(zero_tensor3, logit_c3 * (gt_normal - logit_n),
                                          weights=mask_crop3)

    real_n_loss = tf.losses.absolute_difference(gt_normal, logit_n, weights=mask_crop3)

    # depth loss (l2)
    gt_depth = slice_tensor(depth, logit_n)
    d_loss = tf.losses.mean_squared_error(zero_tensor, logit_c * (gt_depth - logit_d),
                                          weights=mask_crop)

    real_d_loss = tf.losses.absolute_difference(gt_depth, logit_d, weights=mask_crop)

    # omega_loss (l2)
    omega_loss = tf.losses.mean_squared_error(zero_tensor, logit_c - 1.0, weights=mask_crop)

    # depth sample loss (l2)
    d_mask_crop = slice_tensor(ds_mask, logit_n)
    ds_loss = tf.losses.mean_squared_error(gt_depth, logit_d, weights=d_mask_crop)

    # regularization loss (l2)
    r_loss = reg_loss(logit_n, logit_d, shape_mask, cl_mask_inverse, fl_mask_inv)

    total_loss = hyper_params['dlossScale'] * d_loss + hyper_params['nlossScale'] * n_loss + omega_loss + \
                 hyper_params['dsWeight'] * ds_loss + hyper_params['regWeight'] * r_loss

    shape_mask_crop = slice_tensor(shape_mask, logit_n)
    shape_mask_crop3 = tf.tile(shape_mask_crop, [1, 1, 1, 3])
    shape_mask_crop4 = tf.tile(shape_mask_crop, [1, 1, 1, 4])
    cl_mask_inverse4 = tf.tile(cl_mask_inverse, [1, 1, 1, 4])
    gt_f = slice_tensor(gt_f, logit_n) * cl_mask_inverse4
    logit_f = logit_f * shape_mask_crop4 * cl_mask_inverse4
    cur_shape = tf.shape(logit_n)
    lc = tf.zeros([cur_shape[0], cur_shape[1], cur_shape[2], 1], tf.float32)
    gt_coeff_a = tf.concat([tf.slice(gt_f, [0, 0, 0, 0], [-1, -1, -1, 2]), lc], axis=3)
    gt_coeff_b = tf.concat([tf.slice(gt_f, [0, 0, 0, 2], [-1, -1, -1, 2]), lc], axis=3)
    f_coeff_a = tf.concat([tf.slice(logit_f, [0, 0, 0, 0], [-1, -1, -1, 2]), lc], axis=3)
    f_coeff_b = tf.concat([tf.slice(logit_f, [0, 0, 0, 2], [-1, -1, -1, 2]), lc], axis=3)

    return total_loss, d_loss, n_loss, ds_loss, r_loss, real_d_loss, real_n_loss, omega_loss, \
           gt_normal * shape_mask_crop3, logit_n * shape_mask_crop3, gt_depth * shape_mask_crop, \
           logit_d * shape_mask_crop, gt_ds * shape_mask_crop, npr, \
           slice_tensor(cl_mask_inverse, logit_n) * mask_crop, \
           logit_c * shape_mask_crop, gt_coeff_a, gt_coeff_b, f_coeff_a, f_coeff_b


# testing process
def test_procedure(net, test_records):
    # Load data
    # print("test_records打印值",test_records)  # 在此获取tfrecords文件：test_records = ['../sampleData/test/test_db_sample.tfrecords']
    reader = SketchReader(tfrecord_list=test_records, raw_size=[256, 256, 25],
                          shuffle=False, num_threads=hyper_params['nbThreads'],
                          batch_size=1, nb_epoch=1)  # 实例化读取器 reader
    raw_input = reader.next_batch()  # tfrecords读取，返回值raw_input为两个tensor [<tf.Tensor 'batch:0' shape=(1, 256, 256, 6) dtype=float32>, <tf.Tensor 'batch:1' shape=(1, 256, 256, 17) dtype=float32>]
    # print("raw_input打印值:", raw_input)
    # 这些不为空的13个返回值就是test_inputList，是一个13个变量组成的列表 其中npr_lines(0) ds(1) fm(3) fm_inv(4)由raw_input第一个向量切分得到
    # gt_normal(7) gt_depth(8) gt_field(11) mask_cline_inv(12) mask_shape(9) mask_ds(10) mask2d(15), selm(5), ndotv(6)由第二个向量切分得到
    npr_lines, ds, _, fm, fm_inv, gt_normal, gt_depth, gt_field, mask_cline_inv, mask_shape, mask_ds, _, _, \
    mask2d, selm, ndotv = net.cook_raw_inputs(raw_input)

    # print("npr_lines打印值", npr_lines)  # Tensor("cook_raw_input/Slice:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("ds打印值", ds)  # Tensor("cook_raw_input/Slice_1:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("fm打印值", fm)  # Tensor("cook_raw_input/Slice_3:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("fm_inv打印值", fm_inv)  # Tensor("cook_raw_input/Slice_4:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("gt_normal打印值", gt_normal)  # Tensor("cook_raw_input/Slice_7:0", shape=(1, 256, 256, 3), dtype=float32)
    # print("gt_depth打印值", gt_depth)  # Tensor("cook_raw_input/Slice_8:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("gt_field打印值", gt_field)  # Tensor("cook_raw_input/Slice_11:0", shape=(1, 256, 256, 4), dtype=float32)
    # print("mask_cline_inv打印值",mask_cline_inv)  # Tensor("cook_raw_input/Slice_12:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("mask_shape打印值", mask_shape)  # Tensor("cook_raw_input/Slice_9:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("mask_ds打印值", mask_ds)  # Tensor("cook_raw_input/Slice_10:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("mask2d打印值", mask2d)  # Tensor("cook_raw_input/Slice_15:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("selm打印值", selm)  # Tensor("cook_raw_input/Slice_5:0", shape=(1, 256, 256, 1), dtype=float32)
    # print("ndotv打印值", ndotv)  # Tensor("cook_raw_input/Slice_6:0", shape=(1, 256, 256, 1), dtype=float32)

    # Network forward
    logit_f, _ = net.load_field_net(nprLine_input,
                                    mask2D_input,
                                    ds_input,
                                    fm_input,
                                    selLineMask_input,
                                    vdotnScalar_input,
                                    hyper_params['rootFt'],
                                    is_training=False)

    logit_d, logit_n, logit_c, _ = net.load_GeomNet(nprLine_input,
                                                    ds_input,
                                                    mask2D_input,
                                                    fm_input,
                                                    selLineMask_input,
                                                    vdotnScalar_input,
                                                    logit_f,
                                                    clineInvMask_input,
                                                    hyper_params['rootFt'],
                                                    is_training=False)
    #print("logit_f打印值", logit_f)
    #print("logit_d打印值", logit_d)
    #print("logit_n打印值", logit_n)
    #print("logit_c打印值", logit_c)

    # Test loss
    test_loss, test_d_loss, test_n_loss, test_ds_loss, test_reg_loss, test_real_dloss, \
    test_real_nloss, test_omega_loss, out_gt_normal, out_f_normal, out_gt_depth, out_f_depth, out_gt_ds, gt_lines, \
    reg_mask, out_cf_map, test_gt_a, test_gt_b, test_f_a, test_f_b \
        = loss(logit_d,
               logit_n,
               logit_c,
               gtNormal_input,
               gtDepth_input,
               maskShape_input,
               maskDs_input,
               clineInvMask_input,
               ds_input,
               nprLine_input,
               logit_f,
               gtField_input,
               fmInv_input)

    return test_loss, test_d_loss, test_n_loss, test_ds_loss, test_reg_loss, test_real_dloss, \
           test_real_nloss, test_omega_loss, out_gt_normal, out_f_normal, out_gt_depth, \
           out_f_depth, out_gt_ds, gt_lines, reg_mask, out_cf_map, test_gt_a, test_gt_b, test_f_a, test_f_b, \
           [npr_lines, ds, fm, fm_inv, gt_normal, gt_depth, gt_field,
            mask_cline_inv, mask_shape, mask_ds, mask2d, selm, ndotv]


def test_net():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # Load network
    net = SKETCHNET()

    # Testing data
    data_records = [item for item in os.listdir(hyper_params['dbTest']) if item.endswith('.tfrecords')]
    test_records = [os.path.join(hyper_params['dbTest'], item) for item in data_records if item.find('test') != -1]

    test_loss, test_d_loss, test_n_loss, test_ds_loss, test_r_loss, test_real_dloss, \
    test_real_nloss, test_omega_loss, test_gt_normal, test_f_normal, test_gt_depth, test_f_depth, test_gt_ds, \
    test_gt_lines, test_reg_mask, test_f_cfmap, test_gt_a, test_gt_b, test_f_a, test_f_b, test_inputList \
        = test_procedure(net, test_records)

    # Saver
    tf_saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True

    with tf.Session(config=config) as sess:
        # initialize
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)  # 对变量进行初始化，变量运行前必须做初始化操作

        # Restore model
        ckpt = tf.train.latest_checkpoint(hyper_params['cktDir'])
        if ckpt:
            tf_saver.restore(sess, ckpt)
            test_logger.info('restore from the checkpoint {}'.format(ckpt))

        # write graph:
        tf.train.write_graph(sess.graph_def,
                             hyper_params['outDir'],
                             hyper_params['graphName'],
                             as_text=True)
        test_logger.info('save graph tp pbtxt, done')

        # Start input enqueue threads
        coord = tf.train.Coordinator()  # 创建一个线程管理器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中

        try:
            titr = 0
            avg_loss = 0.0
            while not coord.should_stop():  # 直到线程管理器结束

                # get real input
                test_real_input = sess.run(test_inputList)  # 13个的向量
                all0tensor1 = tf.zeros([1, 256, 256, 1])
                # a = tf.constant(0.5, dtype=tf.float32, shape=(1, 256, 256, 1))
                all0tensor2 = tf.zeros([1, 256, 256, 3])
                all0tensor3 = tf.zeros([1, 256, 256, 4])
                all0tensor = sess.run(all0tensor1)
                all0tensor_2 = sess.run(all0tensor2)
                all0tensor_3 = sess.run(all0tensor3)
                # print("all0tensort打印值", all0tensor)
                # print("test_inputList打印值", test_inputList)
                # print("test_real_input打印值", test_real_input)
                '''
                t_loss, t_d_loss, t_n_loss, t_ds_loss, t_r_loss, t_real_dloss, t_real_nloss, \
                t_omega_loss, t_gt_normal, t_f_normal, t_gt_depth, t_f_depth, t_gt_ds, t_gt_lines, t_reg_mask, \
                t_f_cfmap, t_gt_a, t_gt_b, t_f_a, t_f_b \
                    = sess.run([test_loss, test_d_loss, test_n_loss, test_ds_loss, test_r_loss,
                                test_real_dloss, test_real_nloss, test_omega_loss, test_gt_normal, test_f_normal,
                                test_gt_depth, test_f_depth, test_gt_ds, test_gt_lines, test_reg_mask, test_f_cfmap,
                                test_gt_a, test_gt_b, test_f_a, test_f_b],
                                feed_dict={'npr_input:0': test_real_input[0],
                                          'ds_input:0': test_real_input[1],
                                          'fLMask_input:0': test_real_input[2],
                                          'fLInvMask_input:0': test_real_input[3],
                                          'gtN_input:0': test_real_input[4],
                                          'gtD_input:0': test_real_input[5],
                                          'gtField_input:0': test_real_input[6],
                                          'clIMask_input:0': test_real_input[7],
                                          'shapeMask_input:0': test_real_input[8],
                                          'dsMask_input:0': test_real_input[9],
                                          '2dMask_input:0': test_real_input[10],
                                          'sLMask_input:0': test_real_input[11],
                                          'curvMag_input:0': test_real_input[12]
                                          })
                    '''
                t_loss, t_d_loss, t_n_loss, t_ds_loss, t_r_loss, t_real_dloss, t_real_nloss, \
                t_omega_loss, t_gt_normal, t_f_normal, t_gt_depth, t_f_depth, t_gt_ds, t_gt_lines, t_reg_mask, \
                t_f_cfmap, t_gt_a, t_gt_b, t_f_a, t_f_b \
                    = sess.run([test_loss, test_d_loss, test_n_loss, test_ds_loss, test_r_loss,
                                test_real_dloss, test_real_nloss, test_omega_loss, test_gt_normal, test_f_normal,
                                test_gt_depth, test_f_depth, test_gt_ds, test_gt_lines, test_reg_mask, test_f_cfmap,
                                test_gt_a, test_gt_b, test_f_a, test_f_b],
                               feed_dict={'npr_input:0': test_real_input[0],
                                          'ds_input:0': test_real_input[1],
                                          'fLMask_input:0': test_real_input[2],
                                          'fLInvMask_input:0': test_real_input[3],
                                          'gtN_input:0': test_real_input[4],
                                          'gtD_input:0': test_real_input[5],
                                          'gtField_input:0': test_real_input[6],
                                          'clIMask_input:0': test_real_input[7],
                                          'shapeMask_input:0': test_real_input[8],
                                          'dsMask_input:0': test_real_input[9],
                                          '2dMask_input:0': test_real_input[10],
                                          'sLMask_input:0': test_real_input[11],
                                          'curvMag_input:0': test_real_input[12]
                                          })
                # print(test_real_input[4].shape())#(1,256,256,3)
                # print(test_real_input[6].shape())#(1,256,256,4)
                # npr_input:离散01
                # ds_input:离散01
                # fLMask_input:离散01
                # fLMask_input:离散01
                # fLInvMask_input:0.5
                # gtN_input:离散01
                # gtD_input:离散01(4通道)
                # gtField_input:离散01
                # clIMask_input:离散01
                # shapeMask_input:离散01
                # dsMask_input:离散01
                # 2dMask_input:离散01
                # sLMask_input:离散01
                # curvMag_input:离散01

                # Record loss
                avg_loss += t_loss
                test_logger.info(
                    'Test case {}, loss: {}, {}, {}, {}, {}, {}, {}, 0.0, {}'.format(titr, t_loss, t_real_dloss,
                                                                                     t_real_nloss, t_d_loss,
                                                                                     t_n_loss,
                                                                                     t_ds_loss, t_r_loss,
                                                                                     t_omega_loss))

                # Write img out 最多只输出200个Sketch的以下九张图
                if titr < 10000:
                    # 3D输出
                    fn1 = os.path.join(out_output_img_dir, 'gt_depth_' + str(titr) + '.jpg')
                    fn2 = os.path.join(out_output_img_dir, 'gt_normal_' + str(titr) + '.jpg')
                    fn3 = os.path.join(out_output_img_dir, 'gt_field_a_' + str(titr) + '.jpg')
                    fn4 = os.path.join(out_output_img_dir, 'gt_field_b_' + str(titr) + '.jpg')

                    fn5 = os.path.join(out_output_img_dir, 'fwd_conf_map_' + str(titr) + '.jpg')
                    fn6 = os.path.join(out_output_img_dir, 'fwd_depth_' + str(titr) + '.jpg')
                    fn7 = os.path.join(out_output_img_dir, 'fwd_normal_' + str(titr) + '.jpg')
                    fn8 = os.path.join(out_output_img_dir, 'fwd_field_a_' + str(titr) + '.jpg')
                    fn9 = os.path.join(out_output_img_dir, 'fwd_field_b_' + str(titr) + '.jpg')

                    # ground_truth深度图
                    out_gt_d = t_gt_depth[0, :, :, :]
                    out_gt_d.astype(np.float32)
                    out_gt_d = out_gt_d * 255
                    out_gt_d = np.flip(out_gt_d, 0)  # 按行翻转
                    cv2.imwrite(fn1, out_gt_d)

                    # ground_truth法线图
                    out_gt_normal = t_gt_normal[0, :, :, :]
                    out_gt_normal = out_gt_normal[:, :, [2, 1, 0]]
                    out_gt_normal.astype(np.float32)
                    out_gt_normal = out_gt_normal * 255
                    out_gt_normal = np.flip(out_gt_normal, 0)
                    cv2.imwrite(fn2, out_gt_normal)

                    # ground_truth流场a
                    out_gt_a = t_gt_a[0, :, :, :]
                    out_gt_a = out_gt_a[:, :, [2, 1, 0]]
                    out_gt_a.astype(np.float32)
                    out_gt_a = out_gt_a * 255
                    out_gt_a = np.flip(out_gt_a, 0)
                    cv2.imwrite(fn3, out_gt_a)

                    # ground_truth流场b
                    out_gt_b = t_gt_b[0, :, :, :]
                    out_gt_b = out_gt_b[:, :, [2, 1, 0]]
                    out_gt_b.astype(np.float32)
                    out_gt_b = out_gt_b * 255
                    out_gt_b = np.flip(out_gt_b, 0)
                    cv2.imwrite(fn4, out_gt_b)

                    # 预测的置信度图
                    out_f_cfmap = t_f_cfmap[0, :, :, :]
                    out_f_cfmap.astype(np.float32)
                    out_f_cfmap = out_f_cfmap * 255
                    out_f_cfmap = np.flip(out_f_cfmap, 0)
                    cv2.imwrite(fn5, out_f_cfmap)

                    # 预测的深度图
                    out_f_d = t_f_depth[0, :, :, :]
                    out_f_d.astype(np.float32)
                    out_f_d = out_f_d * 255
                    out_f_d = np.flip(out_f_d, 0)
                    cv2.imwrite(fn6, out_f_d)

                    # 预测的法线图
                    out_f_normal = t_f_normal[0, :, :, :]
                    out_f_normal = out_f_normal[:, :, [2, 1, 0]]
                    out_f_normal.astype(np.float32)
                    out_f_normal = out_f_normal * 255
                    out_f_normal = np.flip(out_f_normal, 0)
                    cv2.imwrite(fn7, out_f_normal)

                    # 预测流场a
                    out_f_a = t_f_a[0, :, :, :]
                    out_f_a = out_f_a[:, :, [2, 1, 0]]
                    out_f_a.astype(np.float32)
                    out_f_a = out_f_a * 255
                    out_f_a = np.flip(out_f_a, 0)
                    cv2.imwrite(fn8, out_f_a)

                    # 预测流场b
                    out_f_b = t_f_b[0, :, :, :]
                    out_f_b = out_f_b[:, :, [2, 1, 0]]
                    out_f_b.astype(np.float32)
                    out_f_b = out_f_b * 255
                    out_f_b = np.flip(out_f_b, 0)
                    cv2.imwrite(fn9, out_f_b)

                    # 原始数据深度 法线

                    fndep = os.path.join(out_input_img_dir, 'org_gt_depth_' + str(titr) + '.jpg')
                    fnnor = os.path.join(out_input_img_dir, 'org_gt_normal_' + str(titr) + '.jpg')

                    # ground_truth深度图
                    gt_d = test_real_input[5][0, :, :, :]
                    gt_d.astype(np.float32)
                    gt_d = gt_d * 255
                    gt_d = np.flip(gt_d, 0)
                    cv2.imwrite(fndep, gt_d)

                    # ground_truth法线图
                    gt_normal = test_real_input[4][0, :, :, :]
                    gt_normal = gt_normal[:, :, [2, 1, 0]]
                    gt_normal.astype(np.float32)
                    gt_normal = gt_normal * 255
                    gt_normal = np.flip(gt_normal, 0)
                    cv2.imwrite(fnnor, gt_normal)

                    # 2D输入
                    fna = os.path.join(out_input_img_dir, 'npr_' + str(titr) + '.jpg')
                    fnb = os.path.join(out_input_img_dir, 'ds_' + str(titr) + '.jpg')
                    fnc = os.path.join(out_input_img_dir, 'fLMask_' + str(titr) + '.jpg')
                    fnd = os.path.join(out_input_img_dir, 'fLInvMask_' + str(titr) + '.jpg')
                    fne = os.path.join(out_input_img_dir, 'clIMask_' + str(titr) + '.jpg')
                    fnf = os.path.join(out_input_img_dir, 'shapeMask_' + str(titr) + '.jpg')
                    fng = os.path.join(out_input_img_dir, 'dsMask_' + str(titr) + '.jpg')
                    fnh = os.path.join(out_input_img_dir, '2dMask_' + str(titr) + '.jpg')
                    fni = os.path.join(out_input_img_dir, 'sLMask_' + str(titr) + '.jpg')
                    fnj = os.path.join(out_input_img_dir, 'curvMag_' + str(titr) + '.jpg')

                    # npr
                    npr_line = test_real_input[0][0, :, :, :]
                    # npr_line = all0tensor[0, :, :, :]
                    npr_line.astype(np.float32)
                    npr_line = npr_line * 255
                    npr_line = np.flip(npr_line, 0)
                    cv2.imwrite(fna, npr_line)

                    # 深度样本ds
                    ds = test_real_input[1][0, :, :, :]
                    # ds = all0tensor[0, :, :, :]
                    ds.astype(np.float32)
                    ds = ds * 255
                    ds = np.flip(ds, 0)
                    cv2.imwrite(fnb, ds)

                    # 特征掩码fLMask
                    fm = test_real_input[2][0, :, :, :]
                    # fm = all0tensor[0, :, :, :]
                    fm.astype(np.float32)
                    fm = fm * 255
                    fm = np.flip(fm, 0)
                    cv2.imwrite(fnc, fm)

                    # fLInvMask
                    fmi = test_real_input[3][0, :, :, :]
                    fmi.astype(np.float32)
                    fmi = fmi * 255
                    fmi = np.flip(fmi, 0)
                    cv2.imwrite(fnd, fmi)

                    # clIMask
                    mask_cline_inv = test_real_input[7][0, :, :, :]
                    # mask_cline_inv = all0tensor[0, :, :, :]
                    mask_cline_inv.astype(np.float32)
                    mask_cline_inv = mask_cline_inv * 255
                    mask_cline_inv = np.flip(mask_cline_inv, 0)
                    cv2.imwrite(fne, mask_cline_inv)

                    # shapeMask_input
                    shapemask_input = test_real_input[8][0, :, :, :]
                    shapemask_input.astype(np.float32)
                    shapemask_input = shapemask_input * 255
                    shapemask_input = np.flip(shapemask_input, 0)
                    cv2.imwrite(fnf, shapemask_input)

                    # dsMask
                    dsmask = test_real_input[9][0, :, :, :]
                    # dsmask = all0tensor[0, :, :, :]
                    dsmask.astype(np.float32)
                    dsmask = dsmask * 255
                    dsmask = np.flip(dsmask, 0)
                    cv2.imwrite(fng, dsmask)

                    # 轮廓模板2dMask
                    mask2d = test_real_input[10][0, :, :, :]
                    # mask2d = all0tensor[0, :, :, :]
                    mask2d.astype(np.float32)
                    mask2d = mask2d * 255
                    mask2d = np.flip(mask2d, 0)
                    cv2.imwrite(fnh, mask2d)

                    # sLMask
                    selm = test_real_input[11][0, :, :, :]
                    # selm = all0tensor[0, :, :, :]
                    selm.astype(np.float32)
                    selm = selm * 255
                    selm = np.flip(selm, 0)
                    cv2.imwrite(fni, selm)

                    # curvMag
                    ndotv = test_real_input[12][0, :, :, :]
                    # ndotv = all0tensor[0, :, :, :]
                    ndotv.astype(np.float32)
                    ndotv = ndotv * 255
                    ndotv = np.flip(ndotv, 0)
                    cv2.imwrite(fnj, ndotv)

                titr += 1
                if titr % 100 == 0:
                    print('Iteration: {}'.format(titr))

            avg_loss /= titr
            test_logger.info('Finish test model, average loss is: {}'.format(avg_loss))

        except tf.errors.OutOfRangeError:
            print('Test Done.')
        finally:
            coord.request_stop()

        # Finish testing
        coord.join(threads)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cktDir', required=True, help='checkpoint directory', type=str)
    parser.add_argument('--dbTest', required=True, help='test dataset directory', type=str)
    parser.add_argument('--outDir', required=True, help='otuput directory', type=str)
    parser.add_argument('--device', help='GPU device index', type=str, default='0')
    parser.add_argument('--graphName', required=True, help='writen graph name, net.pbtxt', type=str)

    args = parser.parse_args()
    hyper_params['cktDir'] = args.cktDir
    hyper_params['dbTest'] = args.dbTest
    hyper_params['outDir'] = args.outDir
    hyper_params['device'] = args.device
    hyper_params['graphName'] = args.graphName

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = hyper_params['device']

    # out img dir
    out_output_img_dir = os.path.join(hyper_params['outDir'], 'test_output_img')
    make_dir(out_output_img_dir)

    out_input_img_dir = os.path.join(hyper_params['outDir'], 'test_input_img')
    make_dir(out_input_img_dir)

    # Set logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(hyper_params['outDir'], 'log.txt'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Training preparation
    logger.info('---Test preparation: ---')

    # Dump parameters
    dump_params(hyper_params['outDir'], hyper_params)

    # Begin training
    test_net()
