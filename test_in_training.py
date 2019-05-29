# Based on CorrelationFusedGather

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer

SEED = 0

def get_data():
    size=(10,3)
    np.random.seed(SEED)
    lf = tf.convert_to_tensor(np.random.rand(*size).astype(np.float32))
    rf = tf.convert_to_tensor(np.random.rand(*size).astype(np.float32))
    hi = tf.convert_to_tensor(np.random.randint(0, size[0], size=size[0]).astype(np.int32))
    return lf, rf, hi

def composed(sparams, params, indices):
    moved = tf.gather(params, indices)
    corr = tf.reduce_sum(tf.multiply(sparams, moved), axis=1)
    loss = tf.square(corr)
    return loss

lib = Path.home() / 'tensorflow/bazel-bin/tensorflow/core/user_ops/gather_corr.so'
assert lib.exists() 
gather_module = tf.load_op_library(str(lib))
gather_corr = gather_module.gather_corr
from gather_corr_grad import *

def composed_new(sparams, params, indices):
    moved = gather_corr(params, indices)
    corr = tf.reduce_sum(tf.multiply(sparams, moved), axis=1)
    loss = tf.square(corr)
    return loss

def run_example(lf, rf, hi, func, sess, its=100):
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    init = tf.initializers.random_normal(seed=SEED)
    dlf1 = tf.layers.dense(lf, 100, kernel_initializer=init)
    drf1 = tf.layers.dense(rf, 100, kernel_initializer=init)    
    dlf1 = tf.layers.dense(dlf1, 100, kernel_initializer=init)
    drf1 = tf.layers.dense(drf1, 100, kernel_initializer=init)
    dhi1 = hi
    loss1 = func(dlf1, drf1, dhi1)
    opt1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(its):
        d = sess.run({'opt1':opt1, 'loss1':loss1})
        losses.append(d['loss1'])
    return losses

if __name__ == "__main__":
    res = []
    for func in [composed, composed, composed_new]:
        with tf.Session() as sess:
            data = get_data()
            r = run_example(*data, func, sess)
            res.append(r)
    
    prec = [[np.abs(x) for x in [a1 - a2, a1 - a3]] for i, (a1, a2, a3) in enumerate(zip(*res))]
    prec_ok, prec_asked = np.array(prec).sum(axis=2).sum(axis=0)
    
    print(prec_ok, prec_asked)
    print("OK" if np.abs(prec_ok-prec_asked) <= 4.0*prec_ok  else "FAILED")
