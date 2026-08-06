// ---------------------------------------------------------------------------
// torchstrap :: shared scaffolding for the consolidated (R, T) CUDA kernels
//
// This is the analogue of ATen's <ATen/native/cuda/MultiTensorApply.cuh>: it holds
// the launch geometry and the pointer bookkeeping every fused optimizer needs, and
// **no arithmetic whatsoever**. Each optimizer's device math stays in its own .cu
// file, so any one of them can be edited without touching the others, and a parity
// failure is always attributable to the kernel that owns the math.
//
// The geometry is the one thing they genuinely share. `State` consolidates every
// parameter of every replica into one `(R, T)` buffer, so ATen's
// `TensorListMetadata` / `block_to_tensor` / `block_to_chunk` indirection has
// nothing left to do and a 2-D grid replaces it:
//
//     blockIdx.y == replica          blockIdx.x == chunk within that row
//
// Do not add math here, and do not grow this into an "optimizer framework". ATen
// keeps `adam_math` and `sgd_math` in separate translation units for the same
// reason.
// ---------------------------------------------------------------------------

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/cuda/MultiTensorApply.cuh> // kILP/kBlockSize/kChunkSize

namespace torchstrap {

// `blockIdx.y` indexes the replica; gridDim.y is capped at 65535 everywhere.
constexpr int64_t kMaxGridY = 65535;

// Stands in for ATen's `TensorListMetadata`: the consolidated `(R, T)` buffers of
// one optimizer, in that optimizer's own argument order (Adam: param, grad,
// exp_avg, exp_avg_sq, [max_exp_avg_sq]; SGD: param, grad, [momentum_buffer]).
//
// Both strides are carried because `register_vmap`'s `movedim(d, 0)` can hand us a
// strided `(R, T)` view for `bdim != 0`; a kernel checks `stride_t == 1` to decide
// whether ATen's vectorized `load_store` path is available.
template <typename scalar_t, int depth>
struct ConsolidatedBuffers {
  scalar_t* args[depth];
  int64_t stride_r[depth];
  int64_t stride_t[depth];
};

// `srcs` must have at least `depth` entries; entries past `depth` are never read,
// which is how the optional trailing buffers (max_exp_avg_sq, momentum_buffer) are
// passed as null in the instantiation that does not use them.
template <typename scalar_t, int depth>
inline ConsolidatedBuffers<scalar_t, depth> fill_buffers(
    at::Tensor* const* srcs) {
  ConsolidatedBuffers<scalar_t, depth> buf;
  for (int i = 0; i < depth; i++) {
    buf.args[i] = srcs[i]->data_ptr<scalar_t>();
    buf.stride_r[i] = srcs[i]->stride(0);
    buf.stride_t[i] = srcs[i]->stride(1);
  }
  return buf;
}

// ATen's own launch shape (kBlockSize = 512, kChunkSize = 65536), with the replica
// axis folded into the y dimension instead of a metadata table.
inline dim3 consolidated_grid(int64_t R, int64_t T) {
  return dim3(
      static_cast<unsigned>(
          (T + at::native::kChunkSize - 1) / at::native::kChunkSize),
      static_cast<unsigned>(R),
      1);
}

// The two preconditions the grid mapping imposes, checked identically by every op.
inline void check_consolidated(const at::Tensor& params, const char* op) {
  TORCH_CHECK(params.dim() == 2, op, ": params must be a 2-D (R, T) buffer");
  TORCH_CHECK(
      params.size(0) <= kMaxGridY,
      op,
      " maps the replica axis onto blockIdx.y, which is capped at ",
      kMaxGridY,
      "; got R = ",
      params.size(0));
}

// The `(R,)` side inputs are tiny. Normalising them to the compute dtype and to
// contiguity once lets a kernel index them directly; both are no-ops in the normal
// case. Each op keeps its own explicit list of calls -- there is no argument-pack
// helper here on purpose, so the argument order stays readable at the call site.
inline at::Tensor to_hyper(const at::Tensor& t, at::ScalarType dtype) {
  return t.to(dtype).contiguous();
}

inline at::Tensor to_mask(const at::Tensor& t) {
  return t.to(at::kBool).contiguous();
}

} // namespace torchstrap
