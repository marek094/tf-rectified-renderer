import tensorflow as tf
import numpy as np
from pathlib import Path

EAGER = False
DENSE = True
ONLY_OP = True

if EAGER:
    tf.enable_eager_execution()
    tf.executing_eagerly() 

from corr_layer import correlation_layer

def gen_inputs(shape, seed=None):
    np.random.seed(seed)
    arr = np.random.rand(*shape).astype(np.float32)
    if EAGER:
        return tf.convert_to_tensor(arr)
    else:
        return arr

def gen_indices(shape, seed=None):
    np.random.seed(seed)
    arr = np.random.rand(*shape).astype(np.float32)
    arr = arr * 30. - 70.
    if EAGER:
        return  tf.convert_to_tensor(arr)
    else:
        return arr

def composed(sparams, params, indices):
    indices = tf.reshape(indices, indices.shape[:-1])
    moved = tf.gather(params, indices)
    # print(params.shape, indices.shape, moved.shape)
    multi = tf.multiply(sparams, moved)
    corr = tf.reduce_sum(multi, axis=1)
    return corr


def composed_new(sparams, params, indices):
    lib = Path.home() / 'tensorflow/bazel-bin/tensorflow/core/user_ops/gather_ops.so'
    assert lib.exists() 
    gather_module = tf.load_op_library(str(lib))
    gather_corr = gather_module.gather_corr
    import gather_corr_grad 
    corr = gather_corr(sparams, params, indices)
    return corr

def test_inference(shape, channels, args, iters=15):
    print(f'# test: inference, {args.impl}')
    for impl_name in [args.impl]:
        with tf.Session() as sess:
            # define graph
            if not EAGER:
                place_lf = tf.placeholder(tf.float32, shape + channels)
                place_rf = tf.placeholder(tf.float32, shape + channels)
                place_di = tf.placeholder(tf.float32, shape + [1])
                correlation = correlation_layer(place_lf, place_di, place_rf, delta=16, impl=impl_name)
                feeder = lambda lf, rf, di: {place_lf: lf, place_rf: rf, place_di: di}

            sum_time = 0
            for s in range(iters+1):
                lf = gen_inputs(shape + channels, seed=10*s+0)
                rf = gen_inputs(shape + channels, seed=10*s+1)
                di = gen_indices(shape + [1],     seed=10*s+2)
            
                start = timer()
                r = sess.run(correlation, feeder(lf, rf, di))
                time = timer()-start
                if s > 0:
                    sum_time += time
                print(r[0, 50, 1000:1005, -1], r.shape, f"{time:.4}s")

            print(f"{impl_name.upper()}\t{sum_time/iters:}")


def test_training(shape, channels, args, iters=15):
    print(f'# test: training, {args.impl}', shape, channels)
    for impl_name in [args.impl]:
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.Session(config=cfg) as sess:
            # define graph
            if not EAGER:
                init = tf.initializers.random_normal(seed=42)
                if DENSE:
                    place_lf = tf.placeholder(tf.float32, shape + [1])
                    place_rf = tf.placeholder(tf.float32, shape + [1])
                    place_di = tf.placeholder(tf.float32, shape + [1])
                    layer = lambda inputs, filters:  tf.layers.dense(inputs, filters, kernel_initializer=init)
                else:
                    place_lf = tf.placeholder(tf.float32, shape + channels)
                    place_rf = tf.placeholder(tf.float32, shape + channels)
                    place_di = tf.placeholder(tf.float32, shape + [1])
                    layer = lambda inputs, filters:  tf.layers.conv2d(inputs, filters, 3, 
                                padding='same', kernel_initializer=init, activation=tf.nn.leaky_relu)

                # learnable layer
                tensor_lf = layer(place_lf, channels[0])
                tensor_rf = layer(place_rf, channels[0])
                tensor_di = layer(place_di, 1)
                
                if ONLY_OP:
                    tensor_lf = tf.reshape(tensor_lf, [-1, channels[0]])
                    tensor_rf = tf.reshape(tensor_rf, [-1, channels[0]])
                    tensor_di = tf.reshape(tf.cast(tf.floor(tensor_di), tf.int32), [-1, 1])
                    if impl_name[:3] == "our":
                        correlation = composed_new(tensor_lf, tensor_rf, tensor_di)
                    elif impl_name[:2] == "tf":
                        correlation = composed(tensor_lf, tensor_rf, tensor_di)
                    else: 
                        assert False, f"Wrong impl name {impl_name}"
                    print('#', correlation.shape)
                else:
                    correlation = correlation_layer(tensor_lf, tensor_di, tensor_rf, delta=10, impl=impl_name)

                loss = tf.square(tf.reduce_mean(1 - correlation)) / (1.0 if ONLY_OP else 100000.0)
                train = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)
                feeder = lambda lf, rf, di: {place_lf: lf, place_rf: rf, place_di: di}
                sess.run(tf.global_variables_initializer())

            sum_time = 0
            for s in range(iters+1):
                lf = gen_inputs(shape + ([1] if DENSE else channels), seed=10*s+0) - 0.5
                rf = gen_inputs(shape + ([1] if DENSE else channels), seed=10*s+1) - 0.5
                di = gen_indices(shape + [1],     seed=10*s+2)
            
                start = timer()
                _, r = sess.run([train, loss], feeder(lf, rf, di))
                time = timer()-start
                if s > 0:
                    sum_time += time
                print('#', r, r.shape, f"{time:.4}s")

            print(f">\t{impl_name.upper()}\t{sum_time/iters:}")



if __name__ == "__main__":
    """ 
    run all:
    $ for arg in trivial original our_fused our_linear_warp tf_linear_warp
    > do 2>/dev/null python3 test_corr_layer.py $arg
    > done
    """

    from timeit import default_timer as timer
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('impl', default=None)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    shape = [2, 256, 1920]
    channels = [16]

    if not args.train:
        test_inference(shape, channels, args)
    else: 
        test_training(shape, channels, args)
     

