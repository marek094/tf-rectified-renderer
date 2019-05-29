load(
    "//tensorflow:tensorflow.bzl", 
    "tf_custom_op_library", 
    "tf_kernel_library",
)

tf_kernel_library(
    name = "gather_corr_functor",
    prefix = "gather_corr_functor",
)

tf_custom_op_library(
    name = "gather_ops.so",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/user_ops:gather_corr_functor"],
    srcs = ["gather_ops.cc", "gather_corr_op.cc"],
)