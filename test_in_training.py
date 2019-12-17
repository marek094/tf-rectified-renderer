# Based on CorrelationFusedGather

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer

SEED = 0

def get_data_large():
    from test_corr_layer import gen_indices, gen_inputs
    size = [10000]
    lf = gen_inputs(size+[16],  seed=SEED+0)
    rf = gen_inputs(size+[16],  seed=SEED+1)
    hi = tf.cast(gen_indices(size+[1], seed=SEED+2), tf.int32)
    return lf, rf, hi

def get_data():
    size=(10,3)
    np.random.seed(SEED)
    lf = tf.convert_to_tensor(np.random.rand(*size).astype(np.float32))
    rf = tf.convert_to_tensor(np.random.rand(*size).astype(np.float32))
    hi = tf.convert_to_tensor(np.random.randint(0, size[0], size=size[0]).astype(np.int32))
    return lf, rf, hi

def composed(sparams, params, indices):
    moved = tf.gather(params, indices)
    corr = tf.multiply(sparams, moved)
    loss = tf.reduce_sum(corr, axis=1)
    return loss

lib = Path.home() / 'tensorflow/bazel-bin/tensorflow/core/user_ops/gather_ops.so'
assert lib.exists() 
gather_module = tf.load_op_library(str(lib))
gather_corr = gather_module.gather_corr
import gather_corr_grad 

def composed_new(sparams, params, indices):
    corr = gather_corr(sparams, params, indices)
    return corr

def run_example(lf, rf, hi, func, sess, its=1000, units=100, large=False):
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    init = tf.initializers.random_normal(seed=SEED)
    dlf1 = tf.layers.dense(lf, units, kernel_initializer=init)
    drf1 = tf.layers.dense(rf, units, kernel_initializer=init)  
    dhi1 = hi
    loss1 = func(dlf1, drf1, dhi1)
    opt1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
    sess.run(tf.global_variables_initializer())
    if large:
        for i in range(its):
            sess.run({'opt1':opt1, 'loss1':loss1})
    else:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('impl', default=None)
    args = parser.parse_args()    

    if args.impl is not None:
        func = {'original' : composed, 'our_fused': composed_new}[args.impl]
        with tf.Session() as sess:
            data = get_data_large()
            start = timer()
            run_example(*data, func, sess, large=True, its=10)
            print(f"{args.impl.upper()}\t{timer()-start:.5}")

    else:
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
