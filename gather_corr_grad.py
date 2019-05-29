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
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    params_shape = math_ops.to_int32(params_shape)

  # Build appropriately shaped IndexedSlices
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None]


@ops.RegisterGradient("GatherCorrV2")
def _GatherCorrV2Grad(op, grad):
  """Gradient for GatherV2 op."""
  # params can be large, so colocate the shape calculation with it.
  #
  # params can be very large for sparse model, array_ops.shape raises
  # exception on the Windows platform when any dimension is larger than
  # int32. params_shape is not used in optimizer apply_sparse gradients,
  # so it's fine to convert it back to int32 regardless of truncation.
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    params_shape = math_ops.to_int32(params_shape)

  indices = op.inputs[1]
  indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
  axis = op.inputs[2]
  axis_static = tensor_util.constant_value(axis)

  # For axis 0 gathers, build an appropriately shaped IndexedSlices.
  if axis_static == 0:
    if context.executing_eagerly():
      params_tail_shape = params_shape.cpu()[1:]
    else:
      params_tail_shape = params_shape[1:]
    values_shape = array_ops.concat([indices_size, params_tail_shape], 0)
    values = array_ops.reshape(grad, values_shape)
    indices = array_ops.reshape(indices, indices_size)
    return [ops.IndexedSlices(values, indices, params_shape), None, None]

  outer_shape = params_shape[:axis]
  outer_dims = array_ops.size(outer_shape)
  inner_shape = params_shape[axis:][1:]
  inner_dims = array_ops.size(inner_shape)

  outer_axes_indices = math_ops.range(outer_dims)
  inner_axes_indices = math_ops.range(outer_dims + 1,
                                      outer_dims + 1 + inner_dims)

  values_shape = array_ops.concat([outer_shape, indices_size, inner_shape], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, indices_size)

  # We need to sum up every slice `values[..., i, ....]` corresponding to
  # `params[..., indices[i], ...]`. Since `unsorted_segment_sum` does not
  # support an axis parameter, we transpose the gather dimension to the front,
  # then use `unsorted_segment_sum` to build a
  # [gather_axis, outer_axes, inner_axes] tensor with all the gradients
  # affecting each index in `gather_axis` summed up.
  transpose_dims = array_ops.concat(
      [[outer_dims], outer_axes_indices, inner_axes_indices], 0)
  values_transpose = array_ops.transpose(values, transpose_dims)
  num_segments = params_shape[axis]

  params_grad = math_ops.unsorted_segment_sum(values_transpose, indices,
                                              num_segments)

  # Inverts the above transpose by moving dimension 0 back to its original
  # position.
  invert_transpose_dims = array_ops.concat(
      [outer_axes_indices + 1, [0], inner_axes_indices], 0)
  params_grad = array_ops.transpose(params_grad, invert_transpose_dims)
  return [params_grad, None, None]