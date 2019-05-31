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
    # params = tf.stop_gradient(params) 
    moved = tf.gather(params, indices)
    print('@%', sparams.shape, params.shape, moved.shape)
    corr = tf.multiply(sparams, moved)
    loss = tf.square(tf.reduce_sum(corr, axis=1))
    return loss


def composed_new(sparams, params, indices):
    lib = Path.home() / 'tensorflow/bazel-bin/tensorflow/core/user_ops/gather_ops.so'
    assert lib.exists() 
    gather_module = tf.load_op_library(str(lib))
    gather_corr = gather_module.gather_corr
    import gather_corr_grad 

    # params = tf.stop_gradient(params) 
    corr = gather_corr(sparams, params, indices)
    print('@#', sparams.shape, indices.shape, params.shape, corr.shape)
    loss = tf.square(corr)
    return loss

def run_example(lf, rf, hi, func, sess, its=1000):
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    init = tf.initializers.random_normal(seed=SEED)
    dlf1 = tf.layers.dense(lf, 100, kernel_initializer=init)
    drf1 = tf.layers.dense(rf, 100, kernel_initializer=init)    
    # dlf1 = tf.layers.dense(dlf1, 100, kernel_initializer=init)
    # drf1 = tf.layers.dense(drf1, 100, kernel_initializer=init)
    dhi1 = hi
    print("d", drf1.shape, hi.shape)
    loss1 = func(dlf1, drf1, dhi1)
    opt1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(its):
        if i == 0:
            d = sess.run({'loss1':loss1})    
            print(d['loss1'])
        else:        
            d = sess.run({'opt1':opt1, 'loss1':loss1})
        losses.append(d['loss1'])

    return losses

if __name__ == "__main__":
    res = []
    for func in [composed, composed_new, composed]:
        with tf.Session() as sess:
            data = get_data()
            print(sess.run(data[2]))
            r = run_example(*data, func, sess)
            res.append(r)
    
    prec = [[np.abs(x) for x in [a1 - a2, a1 - a3]] for i, (a1, a2, a3) in enumerate(zip(*res))]
    prec_asked, prec_ok = np.array(prec).sum(axis=2).sum(axis=0)
    
    print(prec_ok, prec_asked)
    print("OK" if np.abs(prec_ok-prec_asked) <= 10.0*prec_ok else "FAILED")
