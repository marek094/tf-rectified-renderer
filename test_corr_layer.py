import tensorflow as tf
import numpy as np

from corr_layer import correlation_layer

from tensorflow.contrib.compiler import xla

def gen_inputs(shape, seed=None):
    np.random.seed(seed)
    arr = np.random.rand(*shape).astype(np.float32)
    return  tf.convert_to_tensor(arr)

def gen_indices(shape, seed=None):
    np.random.seed(seed)
    arr = np.random.rand(*shape).astype(np.float32)
    arr = arr * 30. - 70.
    return  tf.convert_to_tensor(arr)

if __name__ == "__main__":
    from timeit import default_timer as timer
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('impl', default=None)
    args = parser.parse_args()


    impl_names = ['trivial', 'original', 'our_fused', 'original_op']
    # impl_names = reversed(impl_names)
    for impl_name in [args.impl] if args.impl else impl_names:
        with tf.Session() as sess:
            s = 0
            shape = [2, 256, 1920]
            channels = [32]
            left = gen_inputs(shape + channels, seed=s+0)
            right = gen_inputs(shape + channels, seed=s+1)
            indices = gen_indices(shape + [1], seed=s+2)

            corr = correlation_layer(left, indices, right, delta=4, impl=impl_name)

            start = timer()
            r = sess.run(corr)
            # print(r[0, 5, 5:10, -1], r.shape)
            print(f"{impl_name.upper()}\t{timer()-start:.5}")

