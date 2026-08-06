// ---------------------------------------------------------------------------
// torchstrap :: fused batched SGD, CUDA backend
//
// Registered on the CUDA dispatch key with TORCH_LIBRARY_IMPL, exactly as ATen's
// FusedSgdKernel.cu is registered for `_fused_sgd_`. The operator is declared in
// csrc/stubs.cpp with TORCH_LIBRARY; this file supplies one backend of it, and is
// compiled in only when setup.py found a usable CUDA toolkit.
//
// NUMERICS: unlike cuda/adam.cu, this is a **port, not an include**. ATen ships
// <ATen/native/cuda/fused_adam_utils.cuh> in the wheel, so Adam's device math can
// be called verbatim -- but there is no `fused_sgd_utils.cuh`: `sgd_math` lives in
// an anonymous namespace inside FusedSgdKernel.cu and is not installed. So the
// device math below is transcribed from that file, and the CLAUDE.md rule for
// csrc/cpu/adam.cpp applies here in full: **keep ATen's source form**. Do not
// rewrite an `a + b * c` as an explicit `fmaf`, do not reassociate, do not tidy.
// The `-O2` / default `-ffp-contract=fast` flags in setup.py are what make the
// compiler reach ATen's contraction decisions from that source, and
// test/optimizer/test_aten_sgd_parity.py is what proves it did.
//
// The grid geometry is shared with the other fused optimizers via
// consolidated.cuh -- `blockIdx.y == replica`, `blockIdx.x == chunk within that
// row` -- which replaces ATen's `multi_tensor_apply` / `TensorListMetadata` layer
// wholesale, because `State` already consolidated every parameter of every replica
// into one `(R, T)` buffer. Nothing arithmetic comes from that header.
//
// Three things differ from ATen, each because an ensemble needs them:
//
//   * `is_first_step` is **per replica**, derived from the `(R,)` `state_steps`
//     counter (the same one Adam carries). ATen has only a scalar host bool, which
//     is wrong here: a replica frozen by EarlyStopping before it ever stepped
//     still has an uninitialised momentum buffer while the others do not.
//   * hyperparameters are `(R,)` vectors rather than host doubles. `r ==
//     blockIdx.y` makes each block-uniform, so each is read once into a register.
//   * `if (!active_mask[r]) return;` is block-uniform, so a frozen replica moves
//     zero bytes and its rows stay bit-identical *by construction*.
//
// Whether the momentum buffer exists is the call-level `depth` (3 vs 2), exactly as
// upstream -- note that ATen's CPU kernel derives its own `has_momentum_buffer`
// from the very same expression (`momentum != 0.0`, FusedSGDKernel.cpp:200), so
// that runtime test is a buffer-existence test, not a per-element optimisation, and
// the faithful translation of it here is `depth`.
//
// A *replica* whose momentum is exactly 0 inside a depth-3 call therefore runs the
// momentum body with momentum == 0, which reduces to `buf = g` then `g = buf` --
// plain SGD, at the cost of writing g into an otherwise-unused buffer row. (It is
// exact rather than merely close: `0 * buf + 1 * g` is `g` for any finite buf, and
// the buffers only ever hold gradient-derived values.) Adding a block-uniform
// `if (momentum != 0)` to skip that write was tried and **breaks bit-exactness**:
// the extra branch changes nvcc's contraction decisions inside the body, on both
// the vectorized and the ragged path. Do not reintroduce it.
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MultiTensorApply.cuh> // kILP/kBlockSize/kChunkSize, is_aligned, load_store
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <optional>

#include "consolidated.cuh" // ConsolidatedBuffers, consolidated_grid, to_hyper

namespace torchstrap {
namespace {

using at::native::kBlockSize;
using at::native::kChunkSize;
using at::native::kILP;

// ATen's own indices (FusedSgdKernel.cu:13-15).
constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kMomentumBufferIdx = 2;

// --------------------------------------------------------------------------
// ATen's `sgd_math` (FusedSgdKernel.cu:17-59), transcribed verbatim.
//
// `grad_scale_ptr` is retained even though torchstrap always passes nullptr, and
// that is NOT dead weight: it is a *kernel parameter*, so nvcc compiles the branch
// without knowing it is never taken, and the conditional write-back to
// `r_args[kGradIdx][ii]` it implies changes register pressure enough to change
// nvcc's contraction decisions further down. Removing it was measured to make
// `momentum * buf + (1 - dampening) * g` contract into an fma where ATen leaves it
// as a multiply and an add -- 1 ulp, and a hard failure of
// test_aten_sgd_parity.py's ragged CUDA rows. This is the CLAUDE.md "keep ATen's
// source form" rule biting: it applies to code that looks removable too.
// --------------------------------------------------------------------------
template <typename scalar_t, typename opmath_t, int depth>
C10_DEVICE __forceinline__ void sgd_math(
    scalar_t r_args[depth][kILP],
    const opmath_t weight_decay,
    const opmath_t momentum,
    const opmath_t lr,
    const opmath_t dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr) {
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    auto p = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    auto g = static_cast<opmath_t>(r_args[kGradIdx][ii]);

    if (grad_scale_ptr) {
      g /= static_cast<opmath_t>(*grad_scale_ptr);
      r_args[kGradIdx][ii] = g;
    }
    if (maximize) {
      g *= -1.0;
    }
    if (weight_decay != 0) {
      g += weight_decay * p;
    }
    if constexpr (depth > 2) {
      const auto momentum_buffer = is_first_step
          ? g
          : (momentum * static_cast<opmath_t>(r_args[kMomentumBufferIdx][ii]) +
             (1 - dampening) * g);
      r_args[kMomentumBufferIdx][ii] = momentum_buffer;

      if (nesterov) {
        g = g + momentum * momentum_buffer;
      } else {
        g = momentum_buffer;
      }
    }
    p -= lr * g;
    r_args[kParamIdx][ii] = p;
  }
}

template <typename scalar_t, int depth>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void sgd_step_kernel(
    // param, grad, [momentum_buffer] -- `sgd_math` order.
    ConsolidatedBuffers<scalar_t, depth> buf,
    const scalar_t* __restrict__ state_steps,
    const scalar_t* __restrict__ lr,
    const scalar_t* __restrict__ momentum,
    const scalar_t* __restrict__ dampening,
    const scalar_t* __restrict__ weight_decay,
    const bool* __restrict__ active_mask,
    const float* __restrict__ grad_scale_ptr,
    const int64_t T,
    const bool nesterov,
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
  const opmath_t momentum_opmath = static_cast<opmath_t>(momentum[r]);
  const opmath_t dampening_opmath = static_cast<opmath_t>(dampening[r]);
  const opmath_t weight_decay_opmath = static_cast<opmath_t>(weight_decay[r]);

  // Per-replica, where ATen has a scalar host bool. `state_steps` has already been
  // bumped by the caller, so the very first update of a replica sees 1.
  const bool is_first_step = static_cast<opmath_t>(state_steps[r]) == opmath_t(1);

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
      sgd_math<scalar_t, opmath_t, depth>(
          r_args,
          weight_decay_opmath,
          momentum_opmath,
          lr_opmath,
          dampening_opmath,
          nesterov,
          maximize,
          is_first_step,
          grad_scale_ptr);
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
        for (int j = 0; j < depth; j++) {
          r_args[j][ii] = live ? args[j][i * buf.stride_t[j]] : scalar_t(0);
        }
      }
      sgd_math<scalar_t, opmath_t, depth>(
          r_args,
          weight_decay_opmath,
          momentum_opmath,
          lr_opmath,
          dampening_opmath,
          nesterov,
          maximize,
          is_first_step,
          grad_scale_ptr);
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

template <typename scalar_t, int depth>
void launch_sgd(
    at::Tensor& params,
    at::Tensor& grads,
    at::Tensor* momentum_buffers, // null iff depth == 2
    const at::Tensor& state_steps,
    const at::Tensor& lr,
    const at::Tensor& momentum,
    const at::Tensor& dampening,
    const at::Tensor& weight_decay,
    const at::Tensor& active_mask,
    bool nesterov,
    bool maximize) {
  at::Tensor* srcs[3] = {&params, &grads, momentum_buffers};
  const auto buf = fill_buffers<scalar_t, depth>(srcs);

  const int64_t T = params.size(1);

  sgd_step_kernel<scalar_t, depth>
      <<<consolidated_grid(params.size(0), T),
         kBlockSize,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          buf,
          state_steps.const_data_ptr<scalar_t>(),
          lr.const_data_ptr<scalar_t>(),
          momentum.const_data_ptr<scalar_t>(),
          dampening.const_data_ptr<scalar_t>(),
          weight_decay.const_data_ptr<scalar_t>(),
          active_mask.const_data_ptr<bool>(),
          // torchstrap never passes a grad scale; see the sgd_math comment for why
          // the parameter exists anyway.
          /*grad_scale_ptr=*/nullptr,
          T,
          nesterov,
          maximize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor sgd_step_cuda(
    at::Tensor params,
    at::Tensor grads,
    std::optional<at::Tensor> momentum_buffers,
    at::Tensor state_steps,
    at::Tensor lr,
    at::Tensor momentum,
    at::Tensor dampening,
    at::Tensor weight_decay,
    at::Tensor active_mask,
    bool nesterov,
    bool maximize) {
  check_consolidated(params, "torchstrap::sgd_step_");
  TORCH_CHECK(
      !nesterov || momentum_buffers.has_value(),
      "nesterov=True requires momentum_buffers");

  const int64_t R = params.size(0);
  const int64_t T = params.size(1);

  // Frozen replicas do not advance their clock. Done with ATen ops on the tiny
  // (R,) vector; the kernel reads the bumped value and derives is_first_step.
  state_steps.add_(active_mask.to(state_steps.scalar_type()));

  if (R == 0 || T == 0) {
    return at::zeros({}, params.options());
  }

  const at::cuda::CUDAGuard device_guard(params.device());

  const auto dtype = params.scalar_type();
  const auto steps_c = to_hyper(state_steps, dtype);
  const auto lr_c = to_hyper(lr, dtype);
  const auto momentum_c = to_hyper(momentum, dtype);
  const auto dampening_c = to_hyper(dampening, dtype);
  const auto wd_c = to_hyper(weight_decay, dtype);
  const auto mask_c = to_mask(active_mask);

  at::Tensor mb;
  if (momentum_buffers.has_value()) {
    mb = *momentum_buffers;
  }
  const bool has_momentum = momentum_buffers.has_value();
  at::Tensor* mb_ptr = has_momentum ? &mb : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "sgd_step_cuda", [&] {
        // depth 3 with a momentum buffer, 2 without -- ATen's convention, and what
        // its `FusedSgdMathFunctor` static_asserts on.
        if (has_momentum) {
          launch_sgd<scalar_t, 3>(
              params, grads, mb_ptr, steps_c, lr_c, momentum_c, dampening_c,
              wd_c, mask_c, nesterov, maximize);
        } else {
          launch_sgd<scalar_t, 2>(
              params, grads, mb_ptr, steps_c, lr_c, momentum_c, dampening_c,
              wd_c, mask_c, nesterov, maximize);
        }
      });

  // Freshly allocated and aliasing nothing: the op's real outputs are its declared
  // in-place mutations, but `torch.func.vmap` rejects a function that returns
  // nothing, and vmap-composability is the point of torchstrap.
  return at::zeros({}, params.options());
}

} // namespace

TORCH_LIBRARY_IMPL(torchstrap, CUDA, m) {
  m.impl("sgd_step_", &sgd_step_cuda);
}

} // namespace torchstrap
