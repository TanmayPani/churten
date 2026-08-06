// ---------------------------------------------------------------------------
// torchstrap :: fused batched Adagrad, CUDA backend
//
// Registered on the CUDA dispatch key with TORCH_LIBRARY_IMPL, exactly as ATen's
// FusedAdagradKernel.cu is registered for `_fused_adagrad_`. The operator is
// declared in csrc/stubs.cpp with TORCH_LIBRARY; this file supplies one backend of
// it, and is compiled in only when setup.py found a usable CUDA toolkit.
//
// Like cuda/adam.cu and unlike cuda/sgd.cu, the device math here is **included, not
// ported**: `<ATen/native/cuda/fused_adagrad_utils.cuh>` ships in the wheel, so
// `at::native::adagrad_math` is called verbatim and this kernel and
// `torch.optim.Adagrad(fused=True)` put the same source through the same compiler.
//
// Including it also drags in an unused `FusedAdagradMathFunctor` (the header is one
// anonymous namespace). Expect an unused-function warning and do NOT "fix" it by
// editing or trimming the include -- the point of the include is that it is
// upstream's file, untouched.
//
// **`adagrad_math` takes its hyperparameters as `const double&`**, so for float
// parameters the final line
//
//     param = param - corrected_lr * grad / (std::sqrt(state_sum) + eps);
//
// promotes to **fp64 on the device**, as does `grad += param * weight_decay`. On a
// consumer card (4070: 1/64 fp64 rate) that is a real cost, and it is not
// negotiable: it *is* the bit-exactness. Do not "optimise" these to float. The
// kernel is memory-bound anyway, and it is only two or three ops per element.
//
// The grid geometry is shared with the other fused optimizers via consolidated.cuh
// -- `blockIdx.y == replica`, `blockIdx.x == chunk within that row` -- which
// replaces ATen's `multi_tensor_apply_for_fused_optimizer` /
// `FusedOptimizerTensorListMetadata` layer wholesale, because `State` already
// consolidated every parameter of every replica into one `(R, T)` buffer. In
// particular `corrected_lr` comes from the `(R,)` `state_steps` in-kernel, where
// ATen reads a per-tensor `state_steps_addresses[tensor_loc]`.
//
// `grad_scale_ptr` and `found_inf_ptr` are always null here, and are kept as real
// *kernel parameters* rather than call-site literals so nvcc compiles their
// branches without knowing they are never taken. That is deliberate: see the
// equivalent note in cuda/sgd.cu, where deleting the corresponding branch silently
// changed nvcc's contraction decisions and broke parity by 1 ulp.
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MultiTensorApply.cuh> // kILP/kBlockSize/kChunkSize, is_aligned, load_store
#include <ATen/native/cuda/fused_adagrad_utils.cuh> // adagrad_math, kParamIdx/kGradIdx/kStateSumIdx
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "consolidated.cuh" // ConsolidatedBuffers, consolidated_grid, to_hyper

namespace torchstrap {
namespace {

using at::native::kBlockSize;
using at::native::kChunkSize;
using at::native::kGradIdx;
using at::native::kILP;
using at::native::kParamIdx;
using at::native::kStateSumIdx;

// ATen's `depth` for Adagrad is hardcoded 3 (param, grad, state_sum) -- there is no
// amsgrad-style variant and hence no template parameter.
constexpr int kDepth = 3;

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void adagrad_step_kernel(
    // param, grad, state_sum -- `adagrad_math` order.
    ConsolidatedBuffers<scalar_t, kDepth> buf,
    const scalar_t* __restrict__ state_steps,
    const scalar_t* __restrict__ lr,
    const scalar_t* __restrict__ lr_decay,
    const scalar_t* __restrict__ weight_decay,
    const scalar_t* __restrict__ eps,
    const bool* __restrict__ active_mask,
    const float* __restrict__ grad_scale_ptr,
    const float* __restrict__ found_inf_ptr,
    const int64_t T,
    const bool maximize) {
  using opmath_t = at::opmath_type<scalar_t>;

  const int64_t r = static_cast<int64_t>(blockIdx.y);
  if (!active_mask[r]) {
    return;
  }
  if (found_inf_ptr && *found_inf_ptr == 1) {
    return;
  }

  const int64_t chunk_offset = static_cast<int64_t>(blockIdx.x) * kChunkSize;
  // Elements of this row still ahead of us; may exceed kChunkSize, in which case
  // the loops stop at the chunk boundary, as ATen's do.
  const int64_t n = T - chunk_offset;
  if (n <= 0) {
    return;
  }

  // Per-replica, and in double because that is the type `adagrad_math` takes.
  const double lr_double = static_cast<double>(lr[r]);
  const double lr_decay_double = static_cast<double>(lr_decay[r]);
  const double weight_decay_double = static_cast<double>(weight_decay[r]);
  const double eps_double = static_cast<double>(eps[r]);

  // ATen's own expression (fused_adagrad_utils.cuh:73-79), reading the step from
  // our `(R,)` counter instead of `tl.state_steps_addresses[tensor_loc]`. The step
  // has already been bumped by the caller, so the first update sees 1 and `denom`
  // is exactly 1 -- ATen's `step - 1`.
  const auto corrected_lr = [&]() -> double {
    const auto step_count = static_cast<float>(state_steps[r]);
    const auto denom = 1 + (step_count - 1) * lr_decay_double;
    const auto corrected_lr = lr_double / denom;
    return corrected_lr;
  }();

  scalar_t* args[kDepth];
  bool all_aligned = true;
  bool unit_stride = true;
#pragma unroll
  for (int i = 0; i < kDepth; i++) {
    args[i] =
        buf.args[i] + r * buf.stride_r[i] + chunk_offset * buf.stride_t[i];
    all_aligned = all_aligned && at::native::is_aligned(args[i]);
    unit_stride = unit_stride && (buf.stride_t[i] == 1);
  }

  scalar_t r_args[kDepth][kILP];

  // kChunkSize is a multiple of kILP, so only `n` needs checking.
  if ((n % kILP == 0) && all_aligned && unit_stride) {
    for (int64_t i_start = threadIdx.x;
         i_start * kILP < n && i_start * kILP < kChunkSize;
         i_start += blockDim.x) {
      at::native::load_store(r_args[kParamIdx], args[kParamIdx], 0, i_start);
      at::native::load_store(r_args[kGradIdx], args[kGradIdx], 0, i_start);
      at::native::load_store(r_args[kStateSumIdx], args[kStateSumIdx], 0, i_start);

      at::native::adagrad_math<scalar_t, opmath_t>(
          r_args,
          corrected_lr,
          weight_decay_double,
          eps_double,
          maximize,
          grad_scale_ptr,
          found_inf_ptr);

      at::native::load_store(args[kParamIdx], r_args[kParamIdx], i_start, 0);
      if (grad_scale_ptr) {
        at::native::load_store(args[kGradIdx], r_args[kGradIdx], i_start, 0);
      }
      at::native::load_store(args[kStateSumIdx], r_args[kStateSumIdx], i_start, 0);
    }
  } else {
    // ATen's ragged loop, with `stride_t` multiplied in: its `load_args` /
    // `store_args` hardcode an inner stride of 1, whereas `register_vmap`'s
    // `movedim(d, 0)` can hand us a strided `(R, T)` view for `bdim != 0`. With a
    // unit stride this reduces to exactly ATen's.
    for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
         i_start += blockDim.x * kILP) {
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
        const bool live = (i < n) && (i < kChunkSize);
#pragma unroll
        for (int j = 0; j < kDepth; j++) {
          r_args[j][ii] = live ? args[j][i * buf.stride_t[j]] : scalar_t(0);
        }
      }

      at::native::adagrad_math<scalar_t, opmath_t>(
          r_args,
          corrected_lr,
          weight_decay_double,
          eps_double,
          maximize,
          grad_scale_ptr,
          found_inf_ptr);

#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
        if ((i < n) && (i < kChunkSize)) {
#pragma unroll
          for (int j = 0; j < kDepth; j++) {
            // grad is written back only to hand unscaled gradients to GradScaler,
            // and torchstrap never passes a grad_scale. ATen's own condition.
            if (j != kGradIdx || grad_scale_ptr) {
              args[j][i * buf.stride_t[j]] = r_args[j][ii];
            }
          }
        }
      }
    }
  }
}

template <typename scalar_t>
void launch_adagrad(
    at::Tensor& params,
    at::Tensor& grads,
    at::Tensor& state_sums,
    const at::Tensor& state_steps,
    const at::Tensor& lr,
    const at::Tensor& lr_decay,
    const at::Tensor& weight_decay,
    const at::Tensor& eps,
    const at::Tensor& active_mask,
    bool maximize) {
  at::Tensor* srcs[kDepth] = {&params, &grads, &state_sums};
  const auto buf = fill_buffers<scalar_t, kDepth>(srcs);

  const int64_t T = params.size(1);

  adagrad_step_kernel<scalar_t>
      <<<consolidated_grid(params.size(0), T),
         kBlockSize,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          buf,
          state_steps.const_data_ptr<scalar_t>(),
          lr.const_data_ptr<scalar_t>(),
          lr_decay.const_data_ptr<scalar_t>(),
          weight_decay.const_data_ptr<scalar_t>(),
          eps.const_data_ptr<scalar_t>(),
          active_mask.const_data_ptr<bool>(),
          // Always null; kept as kernel parameters so nvcc cannot see through
          // them. See the header comment.
          /*grad_scale_ptr=*/nullptr,
          /*found_inf_ptr=*/nullptr,
          T,
          maximize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor adagrad_step_cuda(
    at::Tensor params,
    at::Tensor grads,
    at::Tensor state_sums,
    at::Tensor state_steps,
    at::Tensor lr,
    at::Tensor lr_decay,
    at::Tensor weight_decay,
    at::Tensor eps,
    at::Tensor active_mask,
    bool maximize) {
  check_consolidated(params, "torchstrap::adagrad_step_");

  const int64_t R = params.size(0);
  const int64_t T = params.size(1);

  // Frozen replicas do not advance their clock. Done with ATen ops on the tiny
  // (R,) vector; the kernel reads the bumped value and derives `corrected_lr`.
  state_steps.add_(active_mask.to(state_steps.scalar_type()));

  if (R == 0 || T == 0) {
    return at::zeros({}, params.options());
  }

  const at::cuda::CUDAGuard device_guard(params.device());

  const auto dtype = params.scalar_type();
  const auto steps_c = to_hyper(state_steps, dtype);
  const auto lr_c = to_hyper(lr, dtype);
  const auto lr_decay_c = to_hyper(lr_decay, dtype);
  const auto wd_c = to_hyper(weight_decay, dtype);
  const auto eps_c = to_hyper(eps, dtype);
  const auto mask_c = to_mask(active_mask);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "adagrad_step_cuda", [&] {
        launch_adagrad<scalar_t>(
            params, grads, state_sums, steps_c, lr_c, lr_decay_c, wd_c, eps_c,
            mask_c, maximize);
      });

  // Freshly allocated and aliasing nothing: the op's real outputs are its declared
  // in-place mutations, but `torch.func.vmap` rejects a function that returns
  // nothing, and vmap-composability is the point of torchstrap.
  return at::zeros({}, params.options());
}

} // namespace

TORCH_LIBRARY_IMPL(torchstrap, CUDA, m) {
  m.impl("adagrad_step_", &adagrad_step_cuda);
}

} // namespace torchstrap
