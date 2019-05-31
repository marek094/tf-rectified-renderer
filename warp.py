# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image warping using per-pixel flow vectors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops


import tensorflow as tf
from pathlib import Path
lib = Path.home() / 'tensorflow/bazel-bin/tensorflow/core/user_ops/gather_ops.so'
assert lib.exists()
gather_module = tf.load_op_library(str(lib))
gather_corr = gather_module.gather_corr
import sys; sys.path.append('/home/marek/tensorflow/tensorflow/core/user_ops/')
from gather_corr_grad import *

def _interpolate_linear(s_grid, query_points, grid, name='interpolate_linear'):
  """Similar to Matlab's interp2 function.

  Finds values for query points on a grid using bilinear interpolation.

  Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).

  Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`

  Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
  """

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    batch_size, height, width, channels = (array_ops.shape(grid)[0],
                                           array_ops.shape(grid)[1],
                                           array_ops.shape(grid)[2],
                                           array_ops.shape(grid)[3])

    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    with ops.control_dependencies([
        check_ops.assert_equal(
            len(query_points.get_shape()),
            3,
            message='Query points must be 3 dimensional.')
    ]):
      num_queries = height * width

    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            height, 2, message='Grid height must be at least 2.'),
        check_ops.assert_greater_equal(
            width, 2, message='Grid width must be at least 2.')
    ]):
      queries = query_points

    # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
    # is still a valid index into the grid.
    max_floor = math_ops.cast(width - 2, query_type)
    min_floor = constant_op.constant(0.0, dtype=query_type)
    floor = math_ops.minimum(math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
    int_floor = math_ops.cast(floor, dtypes.int32)

    # alpha has the same type as the grid, as we will directly use alpha
    # when taking linear combinations of pixel values from the image.
    alpha = math_ops.cast(queries - floor, grid_type)
    min_alpha = constant_op.constant(0.0, dtype=grid_type)
    max_alpha = constant_op.constant(1.0, dtype=grid_type)
    alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

    # Expand alpha to [b, h, w, 1] so we can use broadcasting
    # (since the alpha values don't depend on the channel).
    alpha = array_ops.expand_dims(alpha, 3)

    flat_shape = [batch_size * height * width, channels]
    flattened_grid = array_ops.reshape(grid, flat_shape)
    batch_offsets = array_ops.reshape(
        math_ops.range(batch_size) * height * width, [batch_size, 1, 1])
    height_offsets = array_ops.reshape(
        math_ops.range(height) * width, [1, height, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(x_coords):
        linear_coordinates = batch_offsets + height_offsets + x_coords
        flattened_coordinates = tf.reshape(linear_coordinates, flat_shape[:1])
        flattened_gathered_values = gather_corr(s_grid, flattened_grid, flattened_coordinates)
        gathered_values = tf.reshape(flattened_gathered_values, queries.shape)
        # print('@', s_grid.shape, flattened_grid.shape, linear_coordinates.shape, flattened_gathered_values.shape, gathered_values.shape)
        return gathered_values

    # grab the pixel values in the 4 corners around each query point
    left = gather(int_floor)
    right = gather(int_floor+1)

    alpha = array_ops.reshape(alpha, left.shape)

    interp = alpha * (right - left) + left

    return interp


def dense_image_warp(image, disp, disp_image, name='warp'):
  """Image warping using per-pixel flow vectors.

  Apply a non-linear warp to the image, where the warp is specified by a dense
  flow field of offset vectors that define the correspondences of pixel values
  in the output image back to locations in the  source image. Specifically, the
  pixel value at output[b, j, i, c] is
  images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

  The locations specified by this formula do not necessarily map to an int
  index. Therefore, the pixel value is obtained by bilinear
  interpolation of the 4 nearest pixels around
  (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
  of the image, we use the nearest pixel values at the image boundary.


  Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).

    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.

  Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
      and same type as input image.

  Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
  """
  # with ops.name_scope(name):
  batch_size, height, width, channels = (array_ops.shape(image)[0],
                                          array_ops.shape(image)[1],
                                          array_ops.shape(image)[2],
                                          array_ops.shape(image)[3])

  
  query_points_on_grid =  disp + array_ops.reshape(math_ops.cast(math_ops.range(width), disp.dtype), [width, 1])
  query_points_flattened = array_ops.reshape(query_points_on_grid, [batch_size, height,  width])
  interpolated = _interpolate_linear(image, query_points_flattened, disp_image)

  return interpolated
