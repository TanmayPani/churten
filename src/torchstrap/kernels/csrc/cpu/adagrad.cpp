// ---------------------------------------------------------------------------
// torchstrap :: fused batched Adagrad, CPU backend
//
// Registered on the CPU dispatch key with TORCH_LIBRARY_IMPL, exactly as ATen's
// FusedAdagradKernel.cpp is registered for `_fused_adagrad_`. The operator is
// declared in csrc/stubs.cpp with TORCH_LIBRARY; this file supplies one backend.
//
// The op mutates `params`, `state_sums` and `state_steps`; each is annotated
// `Tensor(aN!)` with a distinct name in the schema, per the requirement for mutable
// operators. The returned Tensor is a freshly allocated scalar that aliases nothing;
// it exists because `torch.func.vmap` rejects a function returning `()`.
//
// NUMERICS: this is a port of ATen's *CPU* fused Adagrad
// (aten/src/ATen/native/cpu/FusedAdagradKernel.cpp), not of the CUDA `adagrad_math`
// that cuda/adagrad.cu includes verbatim. The guarantee is "bit-identical to
// `torch._fused_adagrad_` on CPU" (see test/optimizer/test_aten_adagrad_parity.py).
//
// **CPU and CUDA Adagrad are much further apart than CPU and CUDA Adam are**, and
// deliberately so. ATen's CUDA `adagrad_math` takes `const double&` hyperparameters,
// so its update line and its weight-decay term run in **fp64 on the device**. ATen's
// CPU kernel does the opposite: it computes `clr = lr / (1 + (step - 1) * lr_decay)`
// on the *host* in double and then narrows it -- `Vec(scalar_t(clr))`,
// `Vec(scalar_t(eps))`, `Vec(scalar_t(weight_decay))` -- so the whole element loop is
// float. Each side is bit-exact against upstream on its own device, which is the
// guarantee a caller can use; the cross-device tolerance in
// test_cpu_cuda_adagrad_parity.py is correspondingly looser than Adam's or SGD's.
// That is expected, not a defect, and must not be "fixed" by widening this file.
//
// As with Adam and SGD, ATen's vectorized body and its `size % Vec::size()` scalar
// tail are not the same expression -- the vec body accumulates `sum_vec` as
// `loadu(state_sum) + grad*grad` while the tail does `state_sum_ptr[d] += ...` and
// then re-reads it. Keep ATen's source form; do not hand-write an `std::fma` and do
// not make the two halves agree.
//
// DELIBERATE DEVIATIONS -- this is a port, not a transcription:
//
//   * ATen splits `adagrad_math` into two SFINAE overloads (Half/BFloat16 converting
//     to float vectors, vs float/double). This file has a single scalar template
//     that accumulates in opmath_t (covering low precision, and serving as the
//     vectorized path's ragged tail) plus one vectorized template for
//     scalar_t == opmath_t. The scalar template follows the *lp* overload's source
//     form, which keeps `state_sum_val` in a register where the float/double
//     overload round-trips it through memory -- for float those are bit-identical
//     (a float store/load is exact), and for bf16/fp16 the register form is both
//     more accurate and what ATen's own lp overload does.
//   * `grad_scale_ptr` is dropped. torchstrap never passes a grad scale, and on CPU
//     the branch has no effect on the surrounding codegen (unlike cuda/sgd.cu, where
//     removing the equivalent branch measurably changed nvcc's contraction choices
//     -- the CPU parity test is what confirms this backend is unaffected).
//
// The parity test is what carries these, not construction.
//
// Per-replica, where ATen has scalars: `lr`, `lr_decay`, `weight_decay` and `eps`
// are `(R,)` vectors, and `clr` is therefore computed per replica in the row
// prologue rather than once per tensor. ATen's runtime `weight_decay != 0` test
// survives as a per-replica test in that prologue, which is where ATen's own scalar
// version effectively sits.
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "consolidated.h" // kChunk, parallel_for_chunks

namespace torchstrap {
namespace {

// Per-replica scalars, hoisted out of the element loop. Every field has already been
// narrowed from ATen's `double` intermediates to the math type, at the same point in
// the computation ATen narrows it (`Vec(scalar_t(clr))`, `Vec(scalar_t(eps))`,
// `Vec(scalar_t(weight_decay))`) -- that placement is load-bearing for
// bit-exactness, so these are precomputed rather than re-derived per element.
template <typename opmath_t>
struct AdagradConsts {
  opmath_t clr; // scalar_t(lr / (1 + (step - 1) * lr_decay)), computed in double
  opmath_t eps;
  opmath_t weight_decay;
  bool has_wd;
  bool maximize;
};

// --------------------------------------------------------------------------
// Scalar path: arbitrary strides, accumulates in opmath_t so bf16/fp16 do their
// arithmetic in float. Also serves as the ragged tail of the vectorized path, which
// is why it mirrors ATen's *scalar tail* expressions rather than its vectorized ones.
// --------------------------------------------------------------------------
template <typename scalar_t, typename opmath_t>
inline void adagrad_span_scalar(
    scalar_t* p, const scalar_t* g, scalar_t* s,
    int64_t sp, int64_t sg, int64_t ss, int64_t n,
    const AdagradConsts<opmath_t>& c) {
  for (int64_t d = 0; d < n; ++d) {
    opmath_t grad_val = static_cast<opmath_t>(g[d * sg]);
    opmath_t param_val = static_cast<opmath_t>(p[d * sp]);

    if (c.maximize) {
      grad_val = -grad_val;
    }
    if (c.has_wd) {
      grad_val += param_val * c.weight_decay;
    }
    opmath_t state_sum_val = static_cast<opmath_t>(s[d * ss]);
    state_sum_val += grad_val * grad_val;
    s[d * ss] = static_cast<scalar_t>(state_sum_val);
    opmath_t std_val = std::sqrt(state_sum_val) + c.eps;
    param_val -= c.clr * grad_val / std_val;
    p[d * sp] = static_cast<scalar_t>(param_val);
  }
}

// --------------------------------------------------------------------------
// Vectorized path: unit inner stride, scalar_t == opmath_t (fp32/fp64 only).
// Line-for-line ATen's float/double `adagrad_math` body.
// --------------------------------------------------------------------------
template <typename scalar_t>
inline void adagrad_span_vec(
    scalar_t* p, const scalar_t* g, scalar_t* s, int64_t n,
    const AdagradConsts<scalar_t>& c) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const int64_t K = Vec::size();

  const Vec vwd(c.weight_decay), veps(c.eps), vclr(c.clr);
  const Vec vneg1(scalar_t(-1.0));

  int64_t d = 0;
  for (; d + K <= n; d += K) {
    Vec param_vec = Vec::loadu(p + d);
    Vec grad_vec = Vec::loadu(g + d);

    if (c.maximize) {
      grad_vec = grad_vec * vneg1;
    }
    if (c.has_wd) {
      grad_vec += param_vec * vwd;
    }

    Vec sum_vec = Vec::loadu(s + d) + grad_vec * grad_vec;
    sum_vec.store(s + d);

    Vec std_vec = sum_vec.sqrt() + veps;
    param_vec = param_vec - vclr * grad_vec / std_vec;
    param_vec.store(p + d);
  }

  if (d < n) {
    adagrad_span_scalar<scalar_t, scalar_t>(
        p + d, g + d, s + d, 1, 1, 1, n - d, c);
  }
}

template <typename scalar_t>
void adagrad_run(
    const at::Tensor& params, const at::Tensor& grads,
    const at::Tensor& state_sums, const at::Tensor& state_steps,
    const at::Tensor& lr, const at::Tensor& lr_decay,
    const at::Tensor& weight_decay, const at::Tensor& eps,
    const at::Tensor& active_mask, bool maximize) {
  using opmath_t = at::opmath_type<scalar_t>;

  const int64_t R = params.size(0);
  const int64_t T = params.size(1);

  auto* p_base = params.data_ptr<scalar_t>();
  const auto* g_base = grads.const_data_ptr<scalar_t>();
  auto* s_base = state_sums.data_ptr<scalar_t>();

  const int64_t p0 = params.stride(0), p1 = params.stride(1);
  const int64_t g0 = grads.stride(0), g1 = grads.stride(1);
  const int64_t s0 = state_sums.stride(0), s1 = state_sums.stride(1);

  // The (R,) side inputs are tiny; make them contiguous once so the row prologue can
  // index them directly. These are no-ops in the normal case.
  const auto steps_c = state_steps.contiguous();
  const auto lr_c = lr.contiguous();
  const auto lrd_c = lr_decay.contiguous();
  const auto wd_c = weight_decay.contiguous();
  const auto eps_c = eps.contiguous();
  const auto mask_c = active_mask.to(at::kBool).contiguous();

  const auto* steps_p = steps_c.const_data_ptr<scalar_t>();
  const auto* lr_p = lr_c.const_data_ptr<scalar_t>();
  const auto* lrd_p = lrd_c.const_data_ptr<scalar_t>();
  const auto* wd_p = wd_c.const_data_ptr<scalar_t>();
  const auto* eps_p = eps_c.const_data_ptr<scalar_t>();
  const auto* mask_p = mask_c.const_data_ptr<bool>();

  // The vectorized path needs unit inner stride everywhere, and needs the storage
  // type to be the math type (so bf16/fp16 take the scalar path, which accumulates
  // in float).
  const bool unit_inner = p1 == 1 && g1 == 1 && s1 == 1;
  const bool can_vec = unit_inner && std::is_same<scalar_t, opmath_t>::value;

  // Frozen replicas are skipped inside the driver (consolidated.h), so nothing of
  // their rows is read or written.
  parallel_for_chunks(R, T, mask_p, [&](int64_t r, int64_t off, int64_t n) {
    // ATen's host-side prologue (FusedAdagradKernel.cpp:155-156), per replica: the
    // corrected learning rate is computed in double and only then narrowed. This is
    // the whole of Adagrad's double-precision content on CPU -- the element loop
    // below is entirely float.
    const double lr_d = static_cast<double>(lr_p[r]);
    const double lr_decay_d = static_cast<double>(lrd_p[r]);
    const double wd_d = static_cast<double>(wd_p[r]);
    const double eps_d = static_cast<double>(eps_p[r]);
    const double step = static_cast<double>(steps_p[r]);
    const double clr = lr_d / (1.0 + (step - 1.0) * lr_decay_d);

    AdagradConsts<opmath_t> c;
    c.clr = static_cast<opmath_t>(clr);
    c.eps = static_cast<opmath_t>(eps_d);
    c.weight_decay = static_cast<opmath_t>(wd_d);
    c.has_wd = wd_d != 0.0;
    c.maximize = maximize;

    scalar_t* p = p_base + r * p0 + off * p1;
    const scalar_t* g = g_base + r * g0 + off * g1;
    scalar_t* s = s_base + r * s0 + off * s1;

    if (can_vec) {
      if constexpr (std::is_same<scalar_t, opmath_t>::value) {
        adagrad_span_vec<scalar_t>(p, g, s, n, c);
      }
    } else {
      adagrad_span_scalar<scalar_t, opmath_t>(p, g, s, p1, g1, s1, n, c);
    }
  });
}

at::Tensor adagrad_step_cpu(
    at::Tensor params, at::Tensor grads, at::Tensor state_sums,
    at::Tensor state_steps, at::Tensor lr, at::Tensor lr_decay,
    at::Tensor weight_decay, at::Tensor eps, at::Tensor active_mask,
    bool maximize) {
  TORCH_CHECK(params.dim() == 2, "params must be a 2-D (R, T) buffer");

  // Frozen replicas do not advance their clock. Done with ATen ops on the tiny (R,)
  // vector; the kernel then reads the bumped value and derives `clr`, so the first
  // update of a replica sees step 1 and ATen's `step - 1` makes `clr == lr`.
  state_steps.add_(active_mask.to(state_steps.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, params.scalar_type(), "adagrad_step_cpu", [&] {
        adagrad_run<scalar_t>(
            params, grads, state_sums, state_steps, lr, lr_decay, weight_decay,
            eps, active_mask, maximize);
      });

  // Freshly allocated, aliases nothing. See the header comment.
  return at::zeros({}, params.options());
}

} // namespace

TORCH_LIBRARY_IMPL(torchstrap, CPU, m) {
  m.impl("adagrad_step_", &adagrad_step_cpu);
}

} // namespace torchstrap
