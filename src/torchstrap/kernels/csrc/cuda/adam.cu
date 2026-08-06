// ---------------------------------------------------------------------------
// torchstrap :: fused batched Adam, CUDA backend
//
// Registered on the CUDA dispatch key with TORCH_LIBRARY_IMPL, exactly as ATen's
// fused_adam.cu is registered for `_fused_adam_`. The operator itself is declared
// in csrc/stubs.cpp with TORCH_LIBRARY; this file supplies one backend of it, and is
// compiled in only when setup.py found a usable CUDA toolkit.
// See https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
//
// The device math is ATen's, not a port of it: `at::native::adam_math` is
// included from <ATen/native/cuda/fused_adam_utils.cuh> and called verbatim, as
// are `load_store`, `is_aligned` and `pow_`. So this kernel and
// `torch.optim.Adam(fused=True)` put the same source through the same compiler,
// which is what makes test_aten_fused_parity.py's `torch.equal` hold. (The
// previous NVRTC build needed an explicit `fmaf` on one line because NVRTC
// declined to contract what nvcc contracts; that workaround is gone.)
//
// What is dropped is ATen's `multi_tensor_apply` / `TensorListMetadata` /
// `block_to_tensor` / `block_to_chunk` layer, which exists to batch many
// separately-allocated tensors into one launch. `State` already solved that by
// consolidating every parameter of every replica into one `(R, T)` buffer, so a
// 2-D grid replaces the whole table:
//
//     blockIdx.y == replica          blockIdx.x == chunk within that row
//
// That geometry -- and only that geometry -- lives in consolidated.cuh, shared
// with the other fused optimizers. Every line of arithmetic below is Adam's own.
//
// The mapping is what makes the three things ATen has no concept of cheap.
// Per-replica hyperparameters are `(R,)` vectors rather than host doubles, and
// `r == blockIdx.y` makes them block-uniform, so each is read once into a
// register. Bias correction is per-replica, from the `(R,)` `state_steps`, in
// kernel. And the freeze test `if (!active_mask[r]) return;` is block-uniform, so
// a frozen replica moves zero bytes and its rows stay bit-identical *by
// construction* -- which is required, not just nice: a replica frozen while its
// step is still 0 has `bias_correction2 == 0`, so `denom` is NaN and
// `lr/bias_correction1` is inf. Those rows must never be written, and an
// arithmetic gate would leak the NaN.
//
// The two access patterns are ATen's. The only change is that the second one
// multiplies in `stride_t`: ATen's `load_args`/`store_args` index element by
// element and assume an inner stride of 1, whereas `register_vmap`'s
// `movedim(d, 0)` can hand us a strided `(R, T)` view for `bdim != 0`. With a
// unit stride it reduces to exactly ATen's loop.
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MultiTensorApply.cuh> // kILP/kBlockSize/kChunkSize, is_aligned, load_store
#include <ATen/native/cuda/Pow.cuh>              // pow_
#include <ATen/native/cuda/fused_adam_utils.cuh> // ADAM_MODE, adam_math, kGradIdx
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <optional>

#include "consolidated.cuh" // kMaxGridY, ConsolidatedBuffers, consolidated_grid

namespace torchstrap {
namespace {

using at::native::ADAM_MODE;
using at::native::kBlockSize;
using at::native::kChunkSize;
using at::native::kGradIdx;
using at::native::kILP;

template <typename scalar_t, int depth, ADAM_MODE adam_mode, bool amsgrad>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void adam_step_kernel(
    // param, grad, exp_avg, exp_avg_sq, [max_exp_avg_sq] -- `adam_math` order.
    ConsolidatedBuffers<scalar_t, depth> buf,
    const scalar_t* __restrict__ state_steps,
    const scalar_t* __restrict__ lr,
    const scalar_t* __restrict__ beta1,
    const scalar_t* __restrict__ beta2,
    const scalar_t* __restrict__ eps,
    const scalar_t* __restrict__ weight_decay,
    const bool* __restrict__ active_mask,
    const int64_t T,
    const bool maximize) {
  using opmath_t = at::opmath_type<scalar_t>;

  const int64_t r = static_cast<int64_t>(blockIdx.y);
  if (!active_mask[r]) {
    return;
  }

  const int64_t chunk_offset = static_cast<int64_t>(blockIdx.x) * kChunkSize;
  // Elements of this row still ahead of us; may exceed kChunkSize, in which case
  // the loops stop at the chunk boundary, as ATen's do.
  const int64_t n = T - chunk_offset;
  if (n <= 0) {
    return;
  }

  const opmath_t lr_opmath = static_cast<opmath_t>(lr[r]);
  const opmath_t beta1_opmath = static_cast<opmath_t>(beta1[r]);
  const opmath_t beta2_opmath = static_cast<opmath_t>(beta2[r]);
  const opmath_t weight_decay_opmath = static_cast<opmath_t>(weight_decay[r]);
  const opmath_t eps_opmath = static_cast<opmath_t>(eps[r]);

  // Bias correction is computed in `opmath_t` (float for float params), NOT in
  // double -- which is exactly what ATen's `FusedAdamMathFunctor` does
  // (fused_adam_utils.cuh: `static_cast<opmath_t>(beta1)`, then
  // `pow_(beta1_opmath, step_count)` returning opmath_t). `state_steps` is likewise
  // read as float, as ATen reads it via `reinterpret_cast<const float*>`.
  //
  // This deliberately differs from csrc/cpu/adam.cpp, which computes the same
  // quantities in **double** because ATen's CPU kernel does ("need to use double
  // here to align with non-fused adam", FusedAdamKernel.cpp). The asymmetry looks
  // like a bug and is not: each backend is bit-identical to ATen *on its own
  // device*, and upstream simply made different choices for the two. Widening this
  // to double would break test_aten_fused_parity.py's CUDA half.
  const opmath_t step_count = static_cast<opmath_t>(state_steps[r]);
  const opmath_t bias_correction1 =
      1 - at::native::pow_(beta1_opmath, step_count);
  const opmath_t bias_correction2 =
      1 - at::native::pow_(beta2_opmath, step_count);
  const opmath_t bias_correction2_sqrt = std::sqrt(bias_correction2);

  scalar_t* args[depth];
  bool all_aligned = true;
  bool unit_stride = true;
#pragma unroll
  for (int i = 0; i < depth; i++) {
    args[i] =
        buf.args[i] + r * buf.stride_r[i] + chunk_offset * buf.stride_t[i];
    all_aligned = all_aligned && at::native::is_aligned(args[i]);
    unit_stride = unit_stride && (buf.stride_t[i] == 1);
  }

  scalar_t r_args[depth][kILP];

  // kChunkSize is a multiple of kILP, so only `n` needs checking.
  if ((n % kILP == 0) && all_aligned && unit_stride) {
    for (int64_t i_start = threadIdx.x;
         i_start * kILP < n && i_start * kILP < kChunkSize;
         i_start += blockDim.x) {
#pragma unroll
      for (int i = 0; i < depth; i++) {
        at::native::load_store(r_args[i], args[i], 0, i_start);
      }
      at::native::adam_math<scalar_t, opmath_t, depth, adam_mode, amsgrad>(
          r_args,
          lr_opmath,
          beta1_opmath,
          beta2_opmath,
          weight_decay_opmath,
          eps_opmath,
          maximize,
          /*grad_scale_ptr=*/nullptr,
          /*found_inf_ptr=*/nullptr,
          bias_correction1,
          bias_correction2_sqrt);
#pragma unroll
      for (int i = 0; i < depth; i++) {
        // grad is read-only: ATen writes it back only to hand unscaled gradients
        // to GradScaler, and torchstrap never passes a grad_scale.
        if (i != kGradIdx) {
          at::native::load_store(args[i], r_args[i], i_start, 0);
        }
      }
    }
  } else {
    for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
         i_start += blockDim.x * kILP) {
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
        const bool live = (i < n) && (i < kChunkSize);
#pragma unroll
        for (int j = 0; j < depth; j++) {
          r_args[j][ii] = live ? args[j][i * buf.stride_t[j]] : scalar_t(0);
        }
      }
      at::native::adam_math<scalar_t, opmath_t, depth, adam_mode, amsgrad>(
          r_args,
          lr_opmath,
          beta1_opmath,
          beta2_opmath,
          weight_decay_opmath,
          eps_opmath,
          maximize,
          /*grad_scale_ptr=*/nullptr,
          /*found_inf_ptr=*/nullptr,
          bias_correction1,
          bias_correction2_sqrt);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        const int64_t i = i_start + threadIdx.x + ii * blockDim.x;
        if ((i < n) && (i < kChunkSize)) {
#pragma unroll
          for (int j = 0; j < depth; j++) {
            if (j != kGradIdx) {
              args[j][i * buf.stride_t[j]] = r_args[j][ii];
            }
          }
        }
      }
    }
  }
}

template <typename scalar_t, ADAM_MODE adam_mode, bool amsgrad>
void launch_adam(
    at::Tensor& params,
    at::Tensor& grads,
    at::Tensor& exp_avgs,
    at::Tensor& exp_avg_sqs,
    at::Tensor* max_exp_avg_sqs, // null iff !amsgrad
    const at::Tensor& state_steps,
    const at::Tensor& lr,
    const at::Tensor& beta1,
    const at::Tensor& beta2,
    const at::Tensor& eps,
    const at::Tensor& weight_decay,
    const at::Tensor& active_mask,
    bool maximize) {
  // depth 4 for Adam, 5 with AMSGrad -- ATen's convention, and what its
  // `adam_math` static_asserts on.
  constexpr int depth = amsgrad ? 5 : 4;

  at::Tensor* srcs[5] = {
      &params, &grads, &exp_avgs, &exp_avg_sqs, max_exp_avg_sqs};
  const auto buf = fill_buffers<scalar_t, depth>(srcs);

  const int64_t T = params.size(1);

  adam_step_kernel<scalar_t, depth, adam_mode, amsgrad>
      <<<consolidated_grid(params.size(0), T),
         kBlockSize,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          buf,
          state_steps.const_data_ptr<scalar_t>(),
          lr.const_data_ptr<scalar_t>(),
          beta1.const_data_ptr<scalar_t>(),
          beta2.const_data_ptr<scalar_t>(),
          eps.const_data_ptr<scalar_t>(),
          weight_decay.const_data_ptr<scalar_t>(),
          active_mask.const_data_ptr<bool>(),
          T,
          maximize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor adam_step_cuda(
    at::Tensor params,
    at::Tensor grads,
    at::Tensor exp_avgs,
    at::Tensor exp_avg_sqs,
    std::optional<at::Tensor> max_exp_avg_sqs,
    at::Tensor state_steps,
    at::Tensor lr,
    at::Tensor beta1,
    at::Tensor beta2,
    at::Tensor eps,
    at::Tensor weight_decay,
    at::Tensor active_mask,
    bool amsgrad,
    bool maximize,
    bool decoupled_weight_decay) {
  check_consolidated(params, "torchstrap::adam_step_");
  TORCH_CHECK(
      !amsgrad || max_exp_avg_sqs.has_value(),
      "amsgrad=True requires max_exp_avg_sqs");

  const int64_t R = params.size(0);
  const int64_t T = params.size(1);

  // Frozen replicas do not advance their clock. Done with ATen ops on the tiny
  // (R,) vector; the kernel reads the bumped value and derives both bias
  // corrections from it.
  state_steps.add_(active_mask.to(state_steps.scalar_type()));

  if (R == 0 || T == 0) {
    return at::zeros({}, params.options());
  }

  const at::cuda::CUDAGuard device_guard(params.device());

  // See `to_hyper` in consolidated.cuh: the (R,) side inputs are normalised to the
  // compute dtype and to contiguity once, so the kernel can index them directly.
  const auto dtype = params.scalar_type();
  const auto steps_c = to_hyper(state_steps, dtype);
  const auto lr_c = to_hyper(lr, dtype);
  const auto beta1_c = to_hyper(beta1, dtype);
  const auto beta2_c = to_hyper(beta2, dtype);
  const auto eps_c = to_hyper(eps, dtype);
  const auto wd_c = to_hyper(weight_decay, dtype);
  const auto mask_c = to_mask(active_mask);

  at::Tensor mx;
  if (max_exp_avg_sqs.has_value()) {
    mx = *max_exp_avg_sqs;
  }
  at::Tensor* mx_ptr = amsgrad ? &mx : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "adam_step_cuda", [&] {
        if (amsgrad) {
          if (decoupled_weight_decay) {
            launch_adam<scalar_t, ADAM_MODE::ADAMW, true>(
                params, grads, exp_avgs, exp_avg_sqs, mx_ptr, steps_c, lr_c,
                beta1_c, beta2_c, eps_c, wd_c, mask_c, maximize);
          } else {
            launch_adam<scalar_t, ADAM_MODE::ORIGINAL, true>(
                params, grads, exp_avgs, exp_avg_sqs, mx_ptr, steps_c, lr_c,
                beta1_c, beta2_c, eps_c, wd_c, mask_c, maximize);
          }
        } else {
          if (decoupled_weight_decay) {
            launch_adam<scalar_t, ADAM_MODE::ADAMW, false>(
                params, grads, exp_avgs, exp_avg_sqs, mx_ptr, steps_c, lr_c,
                beta1_c, beta2_c, eps_c, wd_c, mask_c, maximize);
          } else {
            launch_adam<scalar_t, ADAM_MODE::ORIGINAL, false>(
                params, grads, exp_avgs, exp_avg_sqs, mx_ptr, steps_c, lr_c,
                beta1_c, beta2_c, eps_c, wd_c, mask_c, maximize);
          }
        }
      });

  // Freshly allocated and aliasing nothing: the op's real outputs are its
  // declared in-place mutations, but `torch.func.vmap` rejects a function that
  // returns nothing, and vmap-composability is the point of torchstrap.
  return at::zeros({}, params.options());
}

} // namespace

TORCH_LIBRARY_IMPL(torchstrap, CUDA, m) {
  m.impl("adam_step_", &adam_step_cuda);
}

} // namespace torchstrap
