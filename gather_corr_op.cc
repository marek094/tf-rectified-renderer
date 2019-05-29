/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/user_ops/gather_corr_functor.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class GatherCorrOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit GatherCorrOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // GatherV2 added an axis argument. For backwards compatibility with Gather,
    // fall back to axis 0 if the op does not have an axis input.
    int64 axis = 0;
    if (c->num_inputs() == 3) {
      const Tensor& axis_tensor = c->input(2);
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = axis_tensor.scalar<int32>()();
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64>()();
      } else {
        OP_REQUIRES(c, false,
                    errors::InvalidArgument("axis must be int32 or int64."));
      }
    }

    OP_REQUIRES(
        c, axis >= -params.dims() && axis < params.dims(),
        errors::InvalidArgument("Expected axis in the range [", -params.dims(),
                                ", ", params.dims(), "), but got ", axis));
    if (axis < 0) {
      axis = params.dims() + axis;
    }

    // Check that we have enough index space
    const int64 gather_dim_size = params.dim_size(axis);
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, gather_dim_size <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[", axis, "] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", gather_dim_size, " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is params.shape[0:axis] + indices.shape +
    // params.shape[axis + 1:].
    TensorShape result_shape;
    int64 outer_size = 1;
    int64 inner_size = 1;
    for (int i = 0; i < axis; i++) {
      result_shape.AddDim(params.dim_size(i));
      outer_size *= params.dim_size(i);
    }
    result_shape.AppendShape(indices.shape());
    for (int i = axis + 1; i < params.dims(); i++) {
      result_shape.AddDim(params.dim_size(i));
      inner_size *= params.dim_size(i);
    } 

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (N > 0 && outer_size > 0 && inner_size > 0) {
      auto params_flat =
          params.shaped<T, 3>({outer_size, gather_dim_size, inner_size});
      auto indices_flat = indices.flat<Index>();
      auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});
 
      functor::GatherFunctor<Device, T, Index> functor;
      int64 bad_i = functor(c, params_flat, indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0, 
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
    }
  }
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("GatherCorr")                           \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherCorrOp<dev##Device, type, index_type>);    \
  REGISTER_KERNEL_BUILDER(Name("GatherCorrV2")                         \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices")  \
                              .HostMemory("axis"),                     \
                          GatherCorrOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);
TF_CALL_quint16(REGISTER_GATHER_CPU);
TF_CALL_qint16(REGISTER_GATHER_CPU);
TF_CALL_uint32(REGISTER_GATHER_CPU);
TF_CALL_uint64(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU

#if GOOGLE_CUDA

// Registration of the GPU implementations.
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_bool(REGISTER_GATHER_GPU);
TF_CALL_int32(REGISTER_GATHER_GPU);
TF_CALL_int64(REGISTER_GATHER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_GPU);
TF_CALL_complex64(REGISTER_GATHER_GPU);
TF_CALL_complex128(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL


}  // namespace tensorflow



#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/strided_slice_op.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

REGISTER_OP("GatherCorr")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Attr("validate_indices: bool = true")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));
      ShapeHandle params_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &params_subshape));
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("GatherCorrV2")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Input("axis: Taxis")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .Attr("Taxis: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle params_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &params_shape));

      ShapeHandle indices_shape = c->input(1);
      ShapeHandle unused_axis_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_axis_shape));
      const Tensor* axis_t = c->input_tensor(2);

      // If axis is unknown, we can only infer that the result is params_rank +
      // indices_rank - 1.
      if (axis_t == nullptr) {
        if (c->RankKnown(params_shape) && c->RankKnown(indices_shape)) {
          c->set_output(0, c->UnknownShapeOfRank(c->Rank(params_shape) +
                                                 c->Rank(indices_shape) - 1));
        } else {
          c->set_output(0, c->UnknownShape());
        }
        return Status::OK();
      }

      // Note, axis can be negative.
      int64 axis = 0;
      if (axis_t->dtype() == DT_INT32) {
        axis = axis_t->scalar<int32>()();
      } else {
        axis = axis_t->scalar<int64>()();
      }

      // Check that params has rank of at least axis + 1.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(
          params_shape, axis < 0 ? -axis : axis + 1, &unused));

      ShapeHandle params_outer_subshape;
      TF_RETURN_IF_ERROR(
          c->Subshape(params_shape, 0, axis, &params_outer_subshape));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(
          c->Concatenate(params_outer_subshape, indices_shape, &out));

      // Slice from axis + 1 to the end of params_shape to collect the inner
      // dimensions of the result. Special case -1 here since -1 + 1 wraps, and
      // we slice from 0 to the end of shape. Subshape() handles all other
      // out-of-bounds checking.
      if (axis != -1) {
        ShapeHandle params_inner_subshape;
        TF_RETURN_IF_ERROR(
            c->Subshape(params_shape, axis + 1, &params_inner_subshape));
        TF_RETURN_IF_ERROR(c->Concatenate(out, params_inner_subshape, &out));
      }

      c->set_output(0, out);
      return Status::OK();
    });


}  // namespace tensorflow
