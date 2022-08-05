
import cv2
import numpy as np
import os
import tensorflow as tf

#用opencv读取10张单通道输入图 并转换为(1,256,256,1)的张量
def read10pics(imgdir):

    fna = os.path.join(imgdir, 'npr.jpg')
    fnb = os.path.join(imgdir, 'ds.jpg')
    fnc = os.path.join(imgdir, 'fLMask.jpg')
    fnd = os.path.join(imgdir, 'fLInvMask.jpg')
    fne = os.path.join(imgdir, 'clIMask.jpg')
    fnf = os.path.join(imgdir, 'shapeMask.jpg')
    fng = os.path.join(imgdir, 'dsMask.jpg')
    fnh = os.path.join(imgdir, '2dMask.jpg')
    fni = os.path.join(imgdir, 'sLMask.jpg')
    fnj = os.path.join(imgdir, 'curvMag.jpg')

    npr = cv2.imread(fna, 0)
    npr = npr[np.newaxis, :, :, np.newaxis]
    npr.astype(np.float32)
    npr = npr / 255
    #npr = np.flip(npr, 0)
    npr = tf.convert_to_tensor(npr)

    ds = cv2.imread(fnb, 0)
    ds = ds[np.newaxis, :, :, np.newaxis]
    ds.astype(np.float32)
    ds = ds / 255
    #ds = np.flip(ds, 0)
    ds = tf.convert_to_tensor(ds)

    fm = cv2.imread(fnc, 0)
    fm = fm[np.newaxis, :, :, np.newaxis]
    fm.astype(np.float32)
    fm = fm / 255
    #fm = np.flip(fm, 0)
    fm = tf.convert_to_tensor(fm)

    fmi = cv2.imread(fnd, 0)
    fmi = fmi[np.newaxis, :, :, np.newaxis]
    fmi.astype(np.float32)
    fmi = fmi / 255
    #fmi = np.flip(fmi, 0)
    fmi = tf.convert_to_tensor(fmi)

    cm = cv2.imread(fne, 0)
    cm = cm[np.newaxis, :, :, np.newaxis]
    cm.astype(np.float32)
    cm = cm / 255
    #cm = np.flip(cm, 0)
    cm = tf.convert_to_tensor(cm)

    sm = cv2.imread(fnf, 0)
    sm = sm[np.newaxis, :, :, np.newaxis]
    sm.astype(np.float32)
    sm = sm / 255
    #sm = np.flip(sm, 0)
    sm = tf.convert_to_tensor(sm)

    dsm = cv2.imread(fng, 0)
    dsm = dsm[np.newaxis, :, :, np.newaxis]
    dsm.astype(np.float32)
    dsm = dsm / 255
    #dsm = np.flip(dsm, 0)
    dsm = tf.convert_to_tensor(dsm)

    msk = cv2.imread(fnh, 0)
    msk = msk[np.newaxis, :, :, np.newaxis]
    msk.astype(np.float32)
    msk = msk / 255
    #msk = np.flip(msk, 0)
    msk = tf.convert_to_tensor(msk)

    slm = cv2.imread(fni, 0)
    slm = slm[np.newaxis, :, :, np.newaxis]
    slm.astype(np.float32)
    slm = slm / 255
    #slm = np.flip(slm, 0)
    slm = tf.convert_to_tensor(slm)

    cur = cv2.imread(fnj, 0)
    cur = cur[np.newaxis, :, :, np.newaxis]
    cur.astype(np.float32)
    cur = cur / 255
    #cur = np.flip(cur, 0)
    cur = tf.convert_to_tensor(cur)

    return npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur
    
#以下是测试这个函数好不好用
#if __name__ == "__main__":
#    read10pics(".\\input_pics")