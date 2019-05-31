import tensorflow as tf
import numpy as np
from pathlib import Path
from warp import dense_image_warp as linear_gather_corr

def correlation_layer(inputs, indices, indexed_inputs, delta, impl='original', name=None):
    """correlation layer"""
    dispatch = {
        'trivial': _impl_trivial,
        'original': _impl_original,
        'original_op': _impl_original_op,
        'gather_nd': _impl_gather_nd,
        'our_fused': _impl_our_fused,
        'our_fused_all': _impl_our_fused_all,
    }
    assert impl in dispatch, f"Param `impl` must be one of {list(dispatch)}"
    layer_implementation = dispatch[impl]
    return layer_implementation(inputs, indices, indexed_inputs, delta, name)


def _impl_trivial(inputs, indices, indexed_inputs, delta, name):
    deltas = np.arange(-delta+1, delta).astype(np.float32)
    nbs = []
    for shift in deltas:
        flow = - tf.concat([tf.zeros_like(indices), indices + shift], axis=3)
        warped_inputs = tf.contrib.image.dense_image_warp(indexed_inputs, flow)
        dot = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        nbs.append(dot)
    correlation_layer = tf.concat(nbs, axis=3)
    return correlation_layer


def _impl_original(inputs, indices, indexed_inputs, delta, name):
    from dense_image_warp import dense_image_warp as sampler
    deltas = np.arange(-delta+1, delta).astype(np.float32)
    nbs = []
    for shift in deltas:
        warped_inputs = sampler(indexed_inputs, indices + shift)
        dot = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        nbs.append(dot)
    correlation_layer = tf.concat(nbs, axis=3)
    return correlation_layer


def _impl_original_op(inputs, indices, indexed_inputs, delta, name):
    from dense_image_warp_load_op import dense_image_warp as sampler
    deltas = np.arange(-delta+1, delta).astype(np.float32)
    nbs = []
    for shift in deltas:
        warped_inputs = sampler(indexed_inputs, indices + shift)
        dot = tf.reduce_sum(tf.multiply(inputs, warped_inputs), axis=3, keepdims=True) 
        nbs.append(dot)
    correlation_layer = tf.concat(nbs, axis=3)
    return correlation_layer


def _impl_gather_nd(inputs, indices, indexed_inputs, delta, name):
    raise NotImplementedError


def _impl_our_fused(inputs, indices, indexed_inputs, delta, name):
    deltas = np.arange(-delta+1, delta).astype(np.float32)
    nbs = [linear_gather_corr(inputs, indices + shift, indexed_inputs) for shift in deltas]
    correlation_layer = tf.stack(nbs, axis=3)
    return correlation_layer

def _impl_our_fused_all(inputs, indices, indexed_inputs, delta, name):
    raise NotImplementedError


