// ---------------------------------------------------------------------------
// torchstrap :: fused batched SGD, CPU backend
//
// Registered on the CPU dispatch key with TORCH_LIBRARY_IMPL, exactly as ATen's
// FusedSGDKernel.cpp is registered for `_fused_sgd_`. The operator is declared in
// csrc/stubs.cpp with TORCH_LIBRARY; this file supplies one backend of it.
//
// The op mutates `params`, `momentum_buffers` and `state_steps`; each is annotated
// `Tensor(aN!)` with a distinct name in the schema, per the requirement for
// mutable operators. The returned Tensor is a freshly allocated scalar that aliases
// nothing; it exists because `torch.func.vmap` rejects a function returning `()`.
//
// NUMERICS: this is a port of ATen's *CPU* fused SGD
// (aten/src/ATen/native/cpu/FusedSGDKernel.cpp), not of the CUDA `sgd_math` that
// cuda/sgd.cu ports. As with Adam, PyTorch gives each backend its own formulation
// and the guarantee here is "bit-identical to `torch._fused_sgd_` on CPU" (see
// test/optimizer/test_aten_sgd_parity.py), while cuda/sgd.cu is bit-identical to it
// on CUDA. The two are therefore *not* bit-equal to each other, deliberately.
//
// ATen's CPU file is not internally self-consistent, and the port reproduces that
// on purpose -- its vectorized body and its `size % Vec::size()` scalar tail differ
// in fma contraction at four separate points:
//
//                   vec body                        scalar tail
//   weight decay    fmadd(param, wd, grad)          grad += param * wd
//   momentum        fmadd(1-damp, grad, buf*mom)    buf*mom + grad*(1-damp)
//   nesterov        fmadd(buf, mom, grad)           grad += buf * mom
//   param update    param += grad * (-lr)           param -= grad * lr
//
// Do not "clean up" an `a + b * c` into an explicit std::fma here, and do not make
// the two halves agree. The arithmetic is written in ATen's source form so that the
// same compiler, under setup.py's `-O2` and default `-ffp-contract=fast`, makes the
// same choices. That is the whole mechanism by which the port comes out bit-exact.
//
// DELIBERATE DEVIATIONS -- this is a port, not a transcription:
//
//   * ATen splits `sgd_math` into two SFINAE overloads: one for Half/BFloat16 that
//     converts to float vectors, one for float/double. This file has a single
//     scalar template that accumulates in opmath_t (covering low precision, and
//     serving as the vectorized path's ragged tail) plus one vectorized template
//     for scalar_t == opmath_t. The scalar template follows the *lp* overload's
//     source form, which keeps `momentum_buf_var` in a register where the
//     float/double overload round-trips it through memory -- for float those are
//     bit-identical (a float store/load is exact), and for bf16/fp16 the register
//     form is both more accurate and what ATen's own lp overload does.
//   * `nesterov` is a template parameter here and a runtime bool upstream.
//
// The parity test is what carries these, not construction. Do not assume a future
// edit inherits the proof.
//
// Per-replica, where ATen has scalars: `lr`, `momentum`, `dampening` and
// `weight_decay` are `(R,)` vectors, and `is_first_step` is derived per replica
// from the `(R,)` `state_steps` counter (ATen has only a scalar host bool -- wrong
// for an ensemble, where a replica frozen before its first step still has an
// uninitialised momentum buffer while the others do not). ATen's runtime
// `weight_decay != 0` test survives as a per-replica test in the row prologue,
// which is exactly where ATen's own scalar version sits.
//
// ATen's `momentum != 0.0` is NOT treated that way, because it is not a per-element
// optimisation: FusedSGDKernel.cpp:200 derives `has_momentum_buffer` from that same
// expression, so it is the buffer-existence test, and the faithful translation is
// the call-level HAS_MOMENTUM. A replica whose momentum is exactly 0 inside a call
// that does carry buffers therefore runs the momentum body with momentum == 0,
// which is exactly plain SGD (`0 * buf + 1 * g` is `g` for any finite buf). This is
// deliberately the same choice cuda/sgd.cu makes, so the two backends agree on the
// buffer contents as well as the params -- and on CUDA the alternative was measured
// to break bit-exactness outright (see that file).
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <type_traits>

#include "consolidated.h" // kChunk, parallel_for_chunks

namespace torchstrap {
namespace {

// Per-replica scalars, hoisted out of the element loop. Every field has already
// been narrowed from ATen's `double` parameters to the math type, at the same point
// in the computation ATen narrows it (`Vec(scalar_t(weight_decay))`,
// `Vec(scalar_t(1 - dampening))`, `Vec(scalar_t(-lr))`) -- that placement is
// load-bearing for bit-exactness, so these are precomputed rather than re-derived.
template <typename opmath_t>
struct SgdConsts {
  opmath_t weight_decay;
  opmath_t momentum;
  opmath_t one_minus_dampening; // scalar_t(1 - dampening)
  opmath_t lr;
  opmath_t neg_lr;              // scalar_t(-lr)
  bool has_wd;
  bool has_momentum;
  bool maximize;
  bool is_first_step;
};

// --------------------------------------------------------------------------
// Scalar path: arbitrary strides, accumulates in opmath_t so bf16/fp16 do their
// arithmetic in float. Also serves as the ragged tail of the vectorized path,
// which is why it mirrors ATen's *scalar tail* expressions rather than its
// vectorized ones.
// --------------------------------------------------------------------------
template <typename scalar_t, typename opmath_t, bool NESTEROV>
inline void sgd_span_scalar(
    scalar_t* p, const scalar_t* g, scalar_t* mb,
    int64_t sp, int64_t sg, int64_t smb, int64_t n,
    const SgdConsts<opmath_t>& c) {
  for (int64_t d = 0; d < n; ++d) {
    opmath_t grad_val = static_cast<opmath_t>(g[d * sg]);
    opmath_t param_val = static_cast<opmath_t>(p[d * sp]);

    if (c.maximize) {
      grad_val = -grad_val;
    }
    if (c.has_wd) {
      grad_val += param_val * c.weight_decay;
    }
    if (c.has_momentum) {
      opmath_t momentum_buf_var = static_cast<opmath_t>(mb[d * smb]);
      if (c.is_first_step) {
        momentum_buf_var = grad_val;
      } else {
        momentum_buf_var = momentum_buf_var * c.momentum +
            grad_val * c.one_minus_dampening;
      }
      mb[d * smb] = static_cast<scalar_t>(momentum_buf_var);
      if constexpr (NESTEROV) {
        grad_val += momentum_buf_var * c.momentum;
      } else {
        grad_val = momentum_buf_var;
      }
    }
    p[d * sp] = static_cast<scalar_t>(param_val - grad_val * c.lr);
  }
}

// --------------------------------------------------------------------------
// Vectorized path: unit inner stride, scalar_t == opmath_t (fp32/fp64 only).
// Line-for-line ATen's float/double `sgd_math` body.
// --------------------------------------------------------------------------
template <typename scalar_t, bool NESTEROV>
inline void sgd_span_vec(
    scalar_t* p, const scalar_t* g, scalar_t* mb, int64_t n,
    const SgdConsts<scalar_t>& c) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const int64_t K = Vec::size();

  const Vec vwd(c.weight_decay), vmom(c.momentum);
  const Vec vone_minus_damp(c.one_minus_dampening);
  const Vec vneg_lr(c.neg_lr), vneg1(scalar_t(-1.0));

  int64_t d = 0;
  for (; d + K <= n; d += K) {
    Vec param_vec = Vec::loadu(p + d);
    Vec grad_vec = Vec::loadu(g + d);

    if (c.maximize) {
      grad_vec = grad_vec * vneg1;
    }
    if (c.has_wd) {
      grad_vec = at::vec::fmadd(param_vec, vwd, grad_vec);
    }
    if (c.has_momentum) {
      Vec momentum_vec;
      if (c.is_first_step) {
        momentum_vec = grad_vec;
      } else {
        momentum_vec = Vec::loadu(mb + d) * vmom;
        momentum_vec = at::vec::fmadd(vone_minus_damp, grad_vec, momentum_vec);
      }
      momentum_vec.store(mb + d);
      if constexpr (NESTEROV) {
        grad_vec = at::vec::fmadd(momentum_vec, vmom, grad_vec);
      } else {
        grad_vec = momentum_vec;
      }
    }
    param_vec += grad_vec * vneg_lr;
    param_vec.store(p + d);
  }

  if (d < n) {
    sgd_span_scalar<scalar_t, scalar_t, NESTEROV>(
        p + d, g + d, c.has_momentum ? mb + d : nullptr, 1, 1, 1, n - d, c);
  }
}

template <typename scalar_t, bool HAS_MOMENTUM, bool NESTEROV>
void sgd_run(
    const at::Tensor& params, const at::Tensor& grads,
    const at::Tensor& momentum_buffers, const at::Tensor& state_steps,
    const at::Tensor& lr, const at::Tensor& momentum,
    const at::Tensor& dampening, const at::Tensor& weight_decay,
    const at::Tensor& active_mask, bool maximize) {
  using opmath_t = at::opmath_type<scalar_t>;

  const int64_t R = params.size(0);
  const int64_t T = params.size(1);

  auto* p_base = params.data_ptr<scalar_t>();
  const auto* g_base = grads.const_data_ptr<scalar_t>();
  scalar_t* mb_base =
      HAS_MOMENTUM ? momentum_buffers.data_ptr<scalar_t>() : nullptr;

  const int64_t p0 = params.stride(0), p1 = params.stride(1);
  const int64_t g0 = grads.stride(0), g1 = grads.stride(1);
  const int64_t b0 = HAS_MOMENTUM ? momentum_buffers.stride(0) : 0;
  const int64_t b1 = HAS_MOMENTUM ? momentum_buffers.stride(1) : 1;

  // The (R,) side inputs are tiny; make them contiguous once so the row prologue
  // can index them directly. These are no-ops in the normal case.
  const auto steps_c = state_steps.contiguous();
  const auto lr_c = lr.contiguous();
  const auto mom_c = momentum.contiguous();
  const auto damp_c = dampening.contiguous();
  const auto wd_c = weight_decay.contiguous();
  const auto mask_c = active_mask.to(at::kBool).contiguous();

  const auto* steps_p = steps_c.const_data_ptr<scalar_t>();
  const auto* lr_p = lr_c.const_data_ptr<scalar_t>();
  const auto* mom_p = mom_c.const_data_ptr<scalar_t>();
  const auto* damp_p = damp_c.const_data_ptr<scalar_t>();
  const auto* wd_p = wd_c.const_data_ptr<scalar_t>();
  const auto* mask_p = mask_c.const_data_ptr<bool>();

  // The vectorized path needs unit inner stride everywhere, and needs the storage
  // type to be the math type (so bf16/fp16 take the scalar path, which accumulates
  // in float).
  const bool unit_inner = p1 == 1 && g1 == 1 && (!HAS_MOMENTUM || b1 == 1);
  const bool can_vec = unit_inner && std::is_same<scalar_t, opmath_t>::value;

  // Frozen replicas are skipped inside the driver (consolidated.h), so nothing of
  // their rows is read or written.
  parallel_for_chunks(R, T, mask_p, [&](int64_t r, int64_t off, int64_t n) {
    // ATen passes these as `double` and narrows at each use site; the narrowing
    // points below are ATen's, not ours.
    const double lr_d = static_cast<double>(lr_p[r]);
    const double momentum_d = static_cast<double>(mom_p[r]);
    const double dampening_d = static_cast<double>(damp_p[r]);
    const double wd_d = static_cast<double>(wd_p[r]);

    SgdConsts<opmath_t> c;
    c.weight_decay = static_cast<opmath_t>(wd_d);
    c.momentum = static_cast<opmath_t>(momentum_d);
    c.one_minus_dampening = static_cast<opmath_t>(1 - dampening_d);
    c.lr = static_cast<opmath_t>(lr_d);
    c.neg_lr = static_cast<opmath_t>(-lr_d);
    c.has_wd = wd_d != 0.0;
    // ATen's runtime `momentum != 0.0` is a *buffer-existence* test, not a
    // per-element optimisation -- FusedSGDKernel.cpp:200 derives
    // `has_momentum_buffer` from that identical expression. Here the buffer's
    // existence is already the call-level HAS_MOMENTUM, so that is the faithful
    // translation. A replica whose momentum is exactly 0 inside a call that does
    // carry buffers runs the momentum body with momentum == 0, which is exactly
    // plain SGD (`0 * buf + 1 * g` is `g` for any finite buf); cuda/sgd.cu does the
    // same, so the two backends agree on the buffer contents as well as the params.
    c.has_momentum = HAS_MOMENTUM;
    c.maximize = maximize;
    // `state_steps` has already been bumped by the caller, so the very first
    // update of a replica sees 1.
    c.is_first_step = static_cast<double>(steps_p[r]) == 1.0;

    scalar_t* p = p_base + r * p0 + off * p1;
    const scalar_t* g = g_base + r * g0 + off * g1;
    scalar_t* mb = HAS_MOMENTUM ? mb_base + r * b0 + off * b1 : nullptr;

    if (can_vec) {
      if constexpr (std::is_same<scalar_t, opmath_t>::value) {
        sgd_span_vec<scalar_t, NESTEROV>(p, g, mb, n, c);
      }
    } else {
      sgd_span_scalar<scalar_t, opmath_t, NESTEROV>(
          p, g, mb, p1, g1, b1, n, c);
    }
  });
}

at::Tensor sgd_step_cpu(
    at::Tensor params, at::Tensor grads,
    std::optional<at::Tensor> momentum_buffers, at::Tensor state_steps,
    at::Tensor lr, at::Tensor momentum, at::Tensor dampening,
    at::Tensor weight_decay, at::Tensor active_mask, bool nesterov,
    bool maximize) {
  TORCH_CHECK(params.dim() == 2, "params must be a 2-D (R, T) buffer");
  TORCH_CHECK(
      !nesterov || momentum_buffers.has_value(),
      "nesterov=True requires momentum_buffers");

  const bool has_momentum = momentum_buffers.has_value();
  const at::Tensor mb = has_momentum ? *momentum_buffers : params;

  // Frozen replicas do not advance their clock. Done with ATen ops on the tiny
  // (R,) vector; the kernel then reads the bumped value and derives is_first_step.
  state_steps.add_(active_mask.to(state_steps.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, params.scalar_type(), "sgd_step_cpu", [&] {
        if (has_momentum) {
          if (nesterov) {
            sgd_run<scalar_t, true, true>(
                params, grads, mb, state_steps, lr, momentum, dampening,
                weight_decay, active_mask, maximize);
          } else {
            sgd_run<scalar_t, true, false>(
                params, grads, mb, state_steps, lr, momentum, dampening,
                weight_decay, active_mask, maximize);
          }
        } else {
          sgd_run<scalar_t, false, false>(
              params, grads, mb, state_steps, lr, momentum, dampening,
              weight_decay, active_mask, maximize);
        }
      });

  // Freshly allocated, aliases nothing. See the header comment.
  return at::zeros({}, params.options());
}

} // namespace

TORCH_LIBRARY_IMPL(torchstrap, CPU, m) {
  m.impl("sgd_step_", &sgd_step_cpu);
}

} // namespace torchstrap
