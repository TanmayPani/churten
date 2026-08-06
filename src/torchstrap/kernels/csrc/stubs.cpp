// ---------------------------------------------------------------------------
// torchstrap :: the operator definition and the `torchstrap.kernels._C` module.
//
// This is the out-of-tree equivalent of the `_fused_adam_` / `_fused_sgd_` /
// `_fused_adagrad_` entries in ATen's native_functions.yaml: it declares the schemas, and each
// kernel claims its dispatch key from its own translation unit with
// TORCH_LIBRARY_IMPL (csrc/{cpu,cuda}/{adam,sgd,adagrad}.{cpp,cu}) the same way ATen's FusedAdamKernel.cpp and fused_adam.cu do.
//
// The module itself is the stub from
// https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html -- it exports
// nothing. Its only job is to give `import torchstrap.kernels._C` something to
// import, which is what dlopens the library and runs the static initializers
// below.
// ---------------------------------------------------------------------------

#include <Python.h>
#include <torch/library.h>

TORCH_LIBRARY(torchstrap, m) {
  // Each mutated argument gets a distinct alias annotation, as the tutorial
  // requires of a mutable operator. The returned Tensor is freshly allocated
  // and aliases nothing -- it exists only because torch.func.vmap rejects a
  // function that returns no Tensor, and vmap-composability is the point of
  // torchstrap.
  m.def("adam_step_("
        "Tensor(a!) params, "
        "Tensor grads, "
        "Tensor(b!) exp_avgs, "
        "Tensor(c!) exp_avg_sqs, "
        "Tensor(d!)? max_exp_avg_sqs, "
        "Tensor(e!) state_steps, "
        "Tensor lr, "
        "Tensor beta1, "
        "Tensor beta2, "
        "Tensor eps, "
        "Tensor weight_decay, "
        "Tensor active_mask, "
        "bool amsgrad, "
        "bool maximize, "
        "bool decoupled_weight_decay"
        ") -> Tensor");

  // `momentum_buffers` is optional for the same reason `max_exp_avg_sqs` is: it
  // is absent in ATen's depth-2 instantiation (SGD without momentum).
  m.def("sgd_step_("
        "Tensor(a!) params, "
        "Tensor grads, "
        "Tensor(b!)? momentum_buffers, "
        "Tensor(c!) state_steps, "
        "Tensor lr, "
        "Tensor momentum, "
        "Tensor dampening, "
        "Tensor weight_decay, "
        "Tensor active_mask, "
        "bool nesterov, "
        "bool maximize"
        ") -> Tensor");

  // `state_sums` is not optional: ATen's Adagrad has no amsgrad-style variant, so
  // its `depth` is hardcoded 3 (param, grad, state_sum).
  m.def("adagrad_step_("
        "Tensor(a!) params, "
        "Tensor grads, "
        "Tensor(b!) state_sums, "
        "Tensor(c!) state_steps, "
        "Tensor lr, "
        "Tensor lr_decay, "
        "Tensor weight_decay, "
        "Tensor eps, "
        "Tensor active_mask, "
        "bool maximize"
        ") -> Tensor");
}

extern "C" {
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C",
      /*m_doc=*/nullptr,
      /*m_size=*/-1,
      /*m_methods=*/nullptr,
  };
  return PyModule_Create(&module_def);
}
}
