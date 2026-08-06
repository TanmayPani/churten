// ---------------------------------------------------------------------------
// torchstrap :: fused batched Adam, CPU backend
//
// Registered on the CPU dispatch key with TORCH_LIBRARY_IMPL, exactly as ATen's
// FusedAdamKernel.cpp is registered for `_fused_adam_`. The operator itself is
// declared in csrc/stubs.cpp with TORCH_LIBRARY; this file supplies one backend of
// it. See https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
//
// The op mutates `params`, `exp_avgs`, `exp_avg_sqs`, `max_exp_avg_sqs` and
// `state_steps`; each is annotated `Tensor(aN!)` with a distinct name in the
// schema, per the tutorial's requirement for mutable operators. The returned
// Tensor is a freshly allocated scalar that aliases nothing -- it is NOT one of
// the mutated inputs, so it does not hit the `torch.compile` incompatibility the
// tutorial warns about. It exists because `torch.func.vmap` rejects a function
// returning `()` outright ("must only return Tensors, got type NoneType"), and
// vmap-composability is the point of torchstrap.
//
// NUMERICS: this is a port of ATen's *CPU* fused Adam
// (aten/src/ATen/native/cpu/FusedAdamKernel.cpp), not of the CUDA
// `adam_math` in fused_adam_utils.cuh that cuda/adam.cu ports. PyTorch itself
// gives each backend its own formulation, and the CPU one is the better of the
// two on CPU:
//
//   * bias corrections and `step_size = lr / bias_correction1` are computed in
//     **double**, which is what keeps `1 - beta2^step` from losing most of its
//     significant digits as beta2 -> 1 (at beta2 = 0.9999 and a small step it is
//     ~2e-4, i.e. ~3.7 decimal digits of cancellation in float32).
//   * `exp_avg` is a **conditional lerp**: it switches formulation at lerp
//     weight 0.5 so the interpolation is computed from whichever endpoint is
//     nearer.
//
// So the guarantee here is "bit-identical to `torch._fused_adam_` /
// `torch._fused_adamw_` on CPU" (see test/optimizer/test_aten_fused_parity.py),
// exactly as cuda/adam.cu is bit-identical to them on CUDA. CPU and CUDA results
// are therefore *not* bit-equal to each other -- they are each equal to the
// platform they run on, which is the guarantee a caller can actually use. (Note
// that ATen's CPU kernel is not internally self-consistent either: its
// vectorized body and its `size % Vec::size()` scalar tail differ in fma
// contraction. The arithmetic below is written in the same source form as ATen's
// so the same compiler makes the same choices; do not "clean up" a `a + b * c`
// into an explicit std::fma here.)
//
// DELIBERATE DEVIATION -- this is a port, not a transcription. ATen's CPU
// `adam_math` templates on <scalar_t, opmath_t, adam_mode> and takes `amsgrad` as
// a **runtime bool**; the templates below carry AMSGRAD as a template parameter
// instead (so the amsgrad branch disappears at compile time and the per-replica
// prologue can be hoisted). test_aten_fused_parity.py asserts `torch::equal`
// against `torch._fused_adam_` on CPU, so the deviation is *proven* not to move a
// single bit -- but do not read this file as line-for-line upstream, and do not
// assume a future edit inherits that proof. The parity test is what carries it.
//
// ATen also splits `adam_math` into two SFINAE overloads (Half/BFloat16 converting
// to float vectors, vs float/double); this file uses one template that accumulates
// in opmath_t instead. Same story: covered by the parity test, not by construction.
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
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
// been narrowed from ATen's `double` intermediates to the math type, at the same
// point in the computation ATen narrows it -- that placement is load-bearing for
// bit-exactness, so these are precomputed rather than re-derived per element.
template <typename opmath_t>
struct AdamConsts {
  opmath_t step_size;         // scalar_t(lr / bias_correction1)
  opmath_t beta2;             // scalar_t(beta2)
  opmath_t exp_avg_coeff;     // scalar_t(1 - beta1)
  opmath_t exp_avg_sq_coeff;  // scalar_t(1 - beta2)
  opmath_t bias_correction2_sqrt;
  opmath_t eps;
  opmath_t weight_decay;      // ADAM_MODE::ORIGINAL
  opmath_t wd_mul;            // scalar_t(1 - lr * weight_decay), ADAMW
  bool has_wd;
  bool maximize;
  bool lerp_small;            // |exp_avg_coeff| < 0.5
};

// --------------------------------------------------------------------------
// Scalar path: arbitrary strides, accumulates in opmath_t so bf16/fp16 do their
// arithmetic in float. Also serves as the ragged tail of the vectorized path,
// which is why it mirrors ATen's *scalar tail* expressions rather than its
// vectorized ones.
// --------------------------------------------------------------------------
template <typename scalar_t, typename opmath_t, bool AMSGRAD, bool ADAMW>
inline void adam_span_scalar(
    scalar_t* p, const scalar_t* g, scalar_t* m, scalar_t* v, scalar_t* mx,
    int64_t sp, int64_t sg, int64_t sm, int64_t sv, int64_t smx, int64_t n,
    const AdamConsts<opmath_t>& c) {
  for (int64_t i = 0; i < n; ++i) {
    opmath_t grad_val = static_cast<opmath_t>(g[i * sg]);
    opmath_t param_val = static_cast<opmath_t>(p[i * sp]);

    if (c.maximize) {
      grad_val = -grad_val;
    }
    if (c.has_wd) {
      if constexpr (!ADAMW) {
        grad_val += param_val * c.weight_decay;
      } else {
        param_val = param_val * c.wd_mul;
      }
    }

    // exp_avg.lerp_(grad, 1 - beta1)
    opmath_t exp_avg_var = static_cast<opmath_t>(m[i * sm]);
    if (c.lerp_small) {
      exp_avg_var = exp_avg_var + c.exp_avg_coeff * (grad_val - exp_avg_var);
    } else {
      exp_avg_var = grad_val - (grad_val - exp_avg_var) *
                                   (opmath_t(1) - c.exp_avg_coeff);
    }
    m[i * sm] = static_cast<scalar_t>(exp_avg_var);

    opmath_t exp_avg_sq_var = static_cast<opmath_t>(v[i * sv]);
    exp_avg_sq_var = exp_avg_sq_var * c.beta2;
    exp_avg_sq_var =
        exp_avg_sq_var + c.exp_avg_sq_coeff * grad_val * grad_val;
    v[i * sv] = static_cast<scalar_t>(exp_avg_sq_var);

    opmath_t denom_val;
    if constexpr (AMSGRAD) {
      // std::max semantics, not fmax -- they differ on NaN.
      opmath_t max_exp_avg_sq_var =
          std::max(static_cast<opmath_t>(mx[i * smx]), exp_avg_sq_var);
      mx[i * smx] = static_cast<scalar_t>(max_exp_avg_sq_var);
      denom_val = std::sqrt(max_exp_avg_sq_var) / c.bias_correction2_sqrt + c.eps;
    } else {
      denom_val = std::sqrt(exp_avg_sq_var) / c.bias_correction2_sqrt + c.eps;
    }

    p[i * sp] = static_cast<scalar_t>(
        param_val - c.step_size * exp_avg_var / denom_val);
  }
}

// --------------------------------------------------------------------------
// Vectorized path: unit inner stride, scalar_t == opmath_t (fp32/fp64 only).
// Line-for-line ATen's float/double `adam_math` body.
// --------------------------------------------------------------------------
template <typename scalar_t, bool AMSGRAD, bool ADAMW>
inline void adam_span_vec(
    scalar_t* p, const scalar_t* g, scalar_t* m, scalar_t* v, scalar_t* mx,
    int64_t n, const AdamConsts<scalar_t>& c) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const int64_t K = Vec::size();

  const Vec vbeta2(c.beta2), vsq_coeff(c.exp_avg_sq_coeff);
  const Vec vbc2s(c.bias_correction2_sqrt), veps(c.eps);
  const Vec vwd(c.weight_decay), vwd_mul(c.wd_mul);
  const Vec vneg_step(-c.step_size), vneg1(scalar_t(-1.0));
  // ATen blends `lerp_weight` against `lerp_weight - 1` under a lane-uniform
  // mask; the coefficient and the base endpoint are the same for every lane, so
  // the blend collapses to this branch.
  const Vec vcoeff(c.lerp_small ? c.exp_avg_coeff
                                : c.exp_avg_coeff - scalar_t(1));

  int64_t i = 0;
  for (; i + K <= n; i += K) {
    Vec param_vec = Vec::loadu(p + i);
    Vec grad_vec = Vec::loadu(g + i);

    if (c.maximize) {
      grad_vec = grad_vec * vneg1;
    }
    if (c.has_wd) {
      if constexpr (!ADAMW) {
        grad_vec += param_vec * vwd;
      } else {
        param_vec = param_vec * vwd_mul;
      }
    }

    // exp_avg.lerp_(grad, 1 - beta1)
    Vec exp_avg_vec = Vec::loadu(m + i);
    const Vec base = c.lerp_small ? exp_avg_vec : grad_vec;
    exp_avg_vec = at::vec::fmadd(vcoeff, grad_vec - exp_avg_vec, base);

    Vec exp_avg_sq_vec =
        Vec::loadu(v + i) * vbeta2 + vsq_coeff * grad_vec * grad_vec;
    exp_avg_vec.store(m + i);
    exp_avg_sq_vec.store(v + i);

    Vec denom_vec;
    if constexpr (AMSGRAD) {
      Vec max_exp_avg_sq_vec =
          at::vec::maximum(Vec::loadu(mx + i), exp_avg_sq_vec);
      max_exp_avg_sq_vec.store(mx + i);
      denom_vec = (max_exp_avg_sq_vec.sqrt() / vbc2s) + veps;
    } else {
      denom_vec = (exp_avg_sq_vec.sqrt() / vbc2s) + veps;
    }

    param_vec = param_vec + vneg_step * exp_avg_vec / denom_vec;
    param_vec.store(p + i);
  }

  if (i < n) {
    adam_span_scalar<scalar_t, scalar_t, AMSGRAD, ADAMW>(
        p + i, g + i, m + i, v + i, AMSGRAD ? mx + i : nullptr, 1, 1, 1, 1, 1,
        n - i, c);
  }
}

template <typename scalar_t, bool AMSGRAD, bool ADAMW>
void adam_run(
    const at::Tensor& params, const at::Tensor& grads, const at::Tensor& exp_avgs,
    const at::Tensor& exp_avg_sqs, const at::Tensor& max_exp_avg_sqs,
    const at::Tensor& state_steps, const at::Tensor& lr, const at::Tensor& beta1,
    const at::Tensor& beta2, const at::Tensor& eps,
    const at::Tensor& weight_decay, const at::Tensor& active_mask,
    bool maximize) {
  using opmath_t = at::opmath_type<scalar_t>;

  const int64_t R = params.size(0);
  const int64_t T = params.size(1);

  auto* p_base = params.data_ptr<scalar_t>();
  const auto* g_base = grads.const_data_ptr<scalar_t>();
  auto* m_base = exp_avgs.data_ptr<scalar_t>();
  auto* v_base = exp_avg_sqs.data_ptr<scalar_t>();
  scalar_t* mx_base =
      AMSGRAD ? max_exp_avg_sqs.data_ptr<scalar_t>() : nullptr;

  const int64_t p0 = params.stride(0), p1 = params.stride(1);
  const int64_t g0 = grads.stride(0), g1 = grads.stride(1);
  const int64_t m0 = exp_avgs.stride(0), m1 = exp_avgs.stride(1);
  const int64_t v0 = exp_avg_sqs.stride(0), v1 = exp_avg_sqs.stride(1);
  const int64_t x0 = AMSGRAD ? max_exp_avg_sqs.stride(0) : 0;
  const int64_t x1 = AMSGRAD ? max_exp_avg_sqs.stride(1) : 1;

  // The (R,) side inputs are tiny; make them contiguous once so the inner loop
  // can index them directly. These are no-ops in the normal case.
  const auto steps_c = state_steps.contiguous();
  const auto lr_c = lr.contiguous();
  const auto b1_c = beta1.contiguous();
  const auto b2_c = beta2.contiguous();
  const auto eps_c = eps.contiguous();
  const auto wd_c = weight_decay.contiguous();
  const auto mask_c = active_mask.to(at::kBool).contiguous();

  const auto* steps_p = steps_c.const_data_ptr<scalar_t>();
  const auto* lr_p = lr_c.const_data_ptr<scalar_t>();
  const auto* b1_p = b1_c.const_data_ptr<scalar_t>();
  const auto* b2_p = b2_c.const_data_ptr<scalar_t>();
  const auto* eps_p = eps_c.const_data_ptr<scalar_t>();
  const auto* wd_p = wd_c.const_data_ptr<scalar_t>();
  const auto* mask_p = mask_c.const_data_ptr<bool>();

  // The vectorized path needs unit inner stride everywhere, and needs the
  // storage type to be the math type (so bf16/fp16 take the scalar path, which
  // accumulates in float).
  const bool unit_inner =
      p1 == 1 && g1 == 1 && m1 == 1 && v1 == 1 && (!AMSGRAD || x1 == 1);
  const bool can_vec =
      unit_inner && std::is_same<scalar_t, opmath_t>::value;

  // Frozen replicas are skipped inside the driver (consolidated.h), so nothing of
  // their rows is read or written.
  parallel_for_chunks(R, T, mask_p, [&](int64_t r, int64_t off, int64_t n) {
    // Everything from here to the AdamConsts fill is ATen's
    // adam_fused_step_impl prologue: double, "to align with non-fused adam".
    // Widening the float32 per-replica hyperparameters to double is not a
    // pretence of extra precision -- it is where the cancellation in
    // `1 - beta^step` is absorbed before it reaches float32.
    const double lr_d = static_cast<double>(lr_p[r]);
    const double beta1_d = static_cast<double>(b1_p[r]);
    const double beta2_d = static_cast<double>(b2_p[r]);
    const double eps_d = static_cast<double>(eps_p[r]);
    const double wd_d = static_cast<double>(wd_p[r]);
    const double step = static_cast<double>(steps_p[r]);

    const double bias_correction1 = 1 - std::pow(beta1_d, step);
    const double bias_correction2 = 1 - std::pow(beta2_d, step);
    const double exp_avg_grad_coefficient = 1 - beta1_d;
    const double exp_avg_sq_grad_coefficient = 1 - beta2_d;
    const double bias_correction2_sqrt = std::sqrt(bias_correction2);
    const double step_size = lr_d / bias_correction1;

    AdamConsts<opmath_t> c;
    c.step_size = static_cast<opmath_t>(step_size);
    c.beta2 = static_cast<opmath_t>(beta2_d);
    c.exp_avg_coeff = static_cast<opmath_t>(exp_avg_grad_coefficient);
    c.exp_avg_sq_coeff = static_cast<opmath_t>(exp_avg_sq_grad_coefficient);
    c.bias_correction2_sqrt = static_cast<opmath_t>(bias_correction2_sqrt);
    c.eps = static_cast<opmath_t>(eps_d);
    c.weight_decay = static_cast<opmath_t>(wd_d);
    c.wd_mul = static_cast<opmath_t>(1 - lr_d * wd_d);
    c.has_wd = wd_d != 0.0;
    c.maximize = maximize;
    c.lerp_small = std::abs(c.exp_avg_coeff) < opmath_t(0.5);

    scalar_t* p = p_base + r * p0 + off * p1;
    const scalar_t* g = g_base + r * g0 + off * g1;
    scalar_t* m = m_base + r * m0 + off * m1;
    scalar_t* v = v_base + r * v0 + off * v1;
    scalar_t* mx = AMSGRAD ? mx_base + r * x0 + off * x1 : nullptr;

    if (can_vec) {
      if constexpr (std::is_same<scalar_t, opmath_t>::value) {
        adam_span_vec<scalar_t, AMSGRAD, ADAMW>(p, g, m, v, mx, n, c);
      }
    } else {
      adam_span_scalar<scalar_t, opmath_t, AMSGRAD, ADAMW>(
          p, g, m, v, mx, p1, g1, m1, v1, x1, n, c);
    }
  });
}

at::Tensor adam_step_cpu(
    at::Tensor params, at::Tensor grads, at::Tensor exp_avgs,
    at::Tensor exp_avg_sqs, std::optional<at::Tensor> max_exp_avg_sqs,
    at::Tensor state_steps, at::Tensor lr, at::Tensor beta1, at::Tensor beta2,
    at::Tensor eps, at::Tensor weight_decay, at::Tensor active_mask,
    bool amsgrad, bool maximize, bool decoupled_weight_decay) {
  TORCH_CHECK(params.dim() == 2, "params must be a 2-D (R, T) buffer");
  TORCH_CHECK(
      !amsgrad || max_exp_avg_sqs.has_value(),
      "amsgrad=True requires max_exp_avg_sqs");

  const at::Tensor mx =
      max_exp_avg_sqs.has_value() ? *max_exp_avg_sqs : params;

  // Frozen replicas do not advance their clock. Done with ATen ops on the tiny
  // (R,) vector; the kernel then reads the bumped value and derives both bias
  // corrections from it, so there are no `beta.pow(step)` temporaries.
  state_steps.add_(active_mask.to(state_steps.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, params.scalar_type(), "adam_step_cpu", [&] {
        if (amsgrad) {
          if (decoupled_weight_decay) {
            adam_run<scalar_t, true, true>(
                params, grads, exp_avgs, exp_avg_sqs, mx, state_steps, lr,
                beta1, beta2, eps, weight_decay, active_mask, maximize);
          } else {
            adam_run<scalar_t, true, false>(
                params, grads, exp_avgs, exp_avg_sqs, mx, state_steps, lr,
                beta1, beta2, eps, weight_decay, active_mask, maximize);
          }
        } else {
          if (decoupled_weight_decay) {
            adam_run<scalar_t, false, true>(
                params, grads, exp_avgs, exp_avg_sqs, mx, state_steps, lr,
                beta1, beta2, eps, weight_decay, active_mask, maximize);
          } else {
            adam_run<scalar_t, false, false>(
                params, grads, exp_avgs, exp_avg_sqs, mx, state_steps, lr,
                beta1, beta2, eps, weight_decay, active_mask, maximize);
          }
        }
      });

  // Freshly allocated, aliases nothing. See the header comment.
  return at::zeros({}, params.options());
}

} // namespace

TORCH_LIBRARY_IMPL(torchstrap, CPU, m) {
  m.impl("adam_step_", &adam_step_cpu);
}

} // namespace torchstrap
