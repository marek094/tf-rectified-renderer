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

#ifndef TENSORFLOW_CORE_KERNELS_gather_corr_functor_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_gather_corr_functor_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/user_ops/gather_corr_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, bool is_axis_zero>
__global__ void GatherCorrOpKernel(const T* s_params, const T* params, const Index* indices, T* out,
                               int64 gather_dim_size, int64 indices_size,
                               int64 slice_size, int64 out_size) {
  for(int i : CudaGridRangeX(out_size)) {
    Index batch_i = 0;
    Index indices_i = 0;
    if (is_axis_zero) {
      indices_i = i;
    } else {
      Index batch_indices_i = i;
      // The batch index into params to use for i.
      batch_i = batch_indices_i / indices_size;
      // The index into indices to use for i.
      indices_i = batch_indices_i - batch_i * indices_size;
      // Index into the current slice in params to use for i.
    }

    // Index into the gather axis to use for i.
    Index gather_i = ldg(indices + indices_i);

    T sum = T(0);
    // Check gather_i is in [0, gather_dim_size).
    // Set indices out of range to zero
    // TODO(fpmc): Log an error for transfer back to host.
    if (FastBoundsCheck(gather_i, gather_dim_size)) {
      // params is a [batch_size, gather_dim_size, slice_size] tensor. Read
      // params[batch_i, gather_i, slice_i] and write it to the i'th position in
      // out.
      Index params_i =
          (batch_i * gather_dim_size + gather_i) * slice_size;
      Index s_params_i =
          (batch_i * gather_dim_size + indices_i) * slice_size;
          
      for (int j = 0; j < slice_size; j++) {
        sum += ldg(params + (params_i + j)) * ldg(s_params + (s_params_i + j));
      }
    }
    out[i] = sum;
  }
}

namespace functor {
template <typename T, typename Index>
struct GatherFunctor<GPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor s_params,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 2>::Tensor out) {
    const GPUDevice& d = ctx->eigen_gpu_device();
    const int64 out_size = out.size();
    if (out_size == 0) {
      // We need a check here since the CPU version does useful error checking
      // work if there are nonempty indices but empty slices, so the kernel is
      // executed in that case.  In the GPU case we don't know how to do error
      // checking, so we skip the loop entirely.
      return -1;
    }
    const bool is_axis_zero = params.dimension(0) == 1;
    const int64 gather_dim_size = params.dimension(1);
    const int64 indices_size = indices.size();
    const int64 slice_size = params.dimension(2);

    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    if (is_axis_zero) {
      // clang-format off
      GatherCorrOpKernel<T, Index, true>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              s_params.data(), params.data(), indices.data(), out.data(), gather_dim_size,
              indices_size, slice_size, out_size);
      // clang-format on
    } else {
      // clang-format off
      GatherCorrOpKernel<T, Index, false>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              s_params.data(), params.data(), indices.data(), out.data(), gather_dim_size,
              indices_size, slice_size, out_size);
      // clang-format on
    }
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indicies out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_gather_corr_functor_GPU_CU_H_
