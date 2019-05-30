from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("GatherCorr")
def _GatherCorrGrad(op, grad):
  """Gradient for Gather op."""
  # params can be large, so colocate the shape calculation with it.
  #
  # params can be very large for sparse model, array_ops.shape raises
  # exception on the Windows platform when any dimension is larger than
  # int32. params_shape is not used in optimizer apply_sparse gradients,
  # so it's fine to convert it back to int32 regardless of truncation.
  params = op.inputs[1]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    params_shape = math_ops.to_int32(params_shape)

  s_params = op.inputs[0]
  indices = op.inputs[2]

  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, [1]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)

  # Both multiplications work due to broadcasting support of tf.multiply:
  # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  moved_params = array_ops.gather(params, indices) 
  s_params_grad = math_ops.multiply(moved_params, values)

  nonmoved_grads = math_ops.multiply(s_params, values) 
  params_grad = ops.IndexedSlices(nonmoved_grads, indices, params_shape) 

  return [s_params_grad, params_grad, None]
