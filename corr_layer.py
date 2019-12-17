import tensorflow as tf
import numpy as np
from pathlib import Path

def correlation_layer(inputs, indices, indexed_inputs, delta, impl='original', name=None):
    """correlation layer"""
    dispatch = {
        'trivial': _impl_trivial,
        'original': _impl_original,
        'original_op': _impl_original_op,
        'gather_nd': _impl_gather_nd,
        'our_fused': _impl_our_fused,
        'our_linear_warp': _impl_our_linear_warp,
        'tf_linear_warp': _impl_tf_linear_warp,
    }
    assert impl in dispatch, f"Param `impl` must be one of {list(dispatch)}"
    assert inputs.shape == indexed_inputs.shape
    assert inputs.shape[:3] == indices.shape[:3]
    assert indices.shape[3] == 1, \
        f"Indices has last dimension: {indices.shape[3]}, ({indices.shape})"

    layer_implementation = dispatch[impl]
    deltas = np.arange(-delta, delta).astype(np.float32)
    return layer_implementation(inputs, indices, indexed_inputs, deltas, name)    

def _impl_trivial(inputs, indices, indexed_inputs, deltas, name):
    nbs = []
    for shift in deltas:
        flow = - tf.concat([tf.zeros_like(indices), indices + shift], axis=3)
        warped_inputs = tf.contrib.image.dense_image_warp(indexed_inputs, flow)
        dot = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        nbs.append(dot)
    correlation_layer = tf.concat(nbs, axis=3)
    return correlation_layer


def _impl_original(inputs, indices, indexed_inputs, deltas, name):
    from dense_image_warp import dense_image_warp as sampler
    nbs = []
    for shift in deltas:
        warped_inputs = sampler(indexed_inputs, indices + shift)
        dot = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        nbs.append(dot)
    correlation_layer = tf.concat(nbs, axis=3)
    return correlation_layer

def _impl_original_op(inputs, indices, indexed_inputs, deltas, name):
    from dense_image_warp_load_op import dense_image_warp as sampler
    nbs = []
    for shift in deltas:
        warped_inputs = sampler(indexed_inputs, indices + shift)
        dot = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        nbs.append(dot)
    correlation_layer = tf.concat(nbs, axis=3)
    return correlation_layer


def _impl_gather_nd(inputs, indices, indexed_inputs, deltas, name):
    raise NotImplementedError


def _impl_our_fused(inputs, indices, indexed_inputs, deltas, name):
    from warp import dense_image_warp as linear_gather_corr
    nbs = [linear_gather_corr(inputs, indices + shift, indexed_inputs) for shift in deltas]
    correlation_layer = tf.stack(nbs, axis=3)
    return correlation_layer

def _impl_tf_linear_warp(*args, **kwargs):
    return _impl_linear_warp(*args, **kwargs, ours=False)

def _impl_our_linear_warp(*args, **kwargs):
    return _impl_linear_warp(*args, **kwargs, ours=True)

def _impl_linear_warp(inputs, indices, indexed_inputs, deltas, name, ours=False):
    # load gather_corr and gradients
    lib = Path.home() / 'tensorflow/bazel-bin/tensorflow/core/user_ops/gather_ops.so'
    assert Path(lib).exists()
    gather_module = tf.load_op_library(str(lib))
    gather_corr = gather_module.gather_corr
    import sys; sys.path.append('/home/marek/tensorflow/tensorflow/core/user_ops/')
    import gather_corr_grad 
    
    def clip(inputs, min_val, max_val):
        min_val = tf.constant(min_val, dtype=inputs.dtype)
        max_val = tf.constant(max_val, dtype=inputs.dtype)
        return tf.minimum(tf.maximum(min_val, inputs), max_val)

    def simple_corr(inputs, int_indices, indexed_inputs):
        b, h, w, c = inputs.shape
        width_offsets  = tf.reshape(tf.range(w),     [1, 1, w])
        height_offsets = tf.reshape(tf.range(h)*w,   [1, h, 1])
        batch_offsets  = tf.reshape(tf.range(b)*h*w, [b, 1, 1])
        int_indices = tf.reshape(int_indices, [b, h, w])
        flatten_indices = clip(width_offsets + int_indices, 0, w-2)
        flatten_indices = batch_offsets + (height_offsets + flatten_indices)
        flatten_indexed_inputs = tf.reshape(indexed_inputs, [b*w*h, c])
        flatten_inputs = tf.reshape(inputs, [b*w*h, c])
        if not ours:
            warped_inputs = tf.gather(flatten_indexed_inputs, tf.reshape(flatten_indices, [b*w*h]))
            warped_inputs = tf.reshape(warped_inputs, [b,h,w,c])
            dots = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        else:
            dots = gather_corr(flatten_inputs, flatten_indexed_inputs, flatten_indices)
        return tf.reshape(dots, [b,h,w,1])

    b, h, w, c = inputs.shape
    floor_indices = tf.floor(indices)
    int_indices = tf.cast(floor_indices, tf.int32)
    # alpha = x-[x]
    alpha = tf.cast(clip(indices - floor_indices, 0.0, 1.0), inputs.dtype)
    alpha = tf.reshape(alpha, [b, h, w, 1])

    ext_deltas = np.append(deltas, deltas[-1]+1).astype(np.int32)
    corrs = [simple_corr(inputs, int_indices + d, indexed_inputs) for d in ext_deltas]

    lower_corr = tf.concat(corrs[:-1], axis=3)
    upper_corr = tf.concat(corrs[1:], axis=3)
    linear_interpolation = alpha * (upper_corr - lower_corr) + lower_corr
    return linear_interpolation
