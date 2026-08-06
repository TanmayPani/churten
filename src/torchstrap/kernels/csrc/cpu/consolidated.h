// ---------------------------------------------------------------------------
// torchstrap :: shared scaffolding for the consolidated (R, T) CPU kernels
//
// The CPU counterpart of cuda/consolidated.cuh, and equally arithmetic-free: it
// holds the chunking constant and the `at::parallel_for` driver, nothing else.
// Every optimizer's element math -- its vectorized body, its scalar tail, and its
// per-replica `double` prologue -- stays in that optimizer's own .cpp, because
// those are exactly the places where ATen's formulations differ per optimizer and
// where bit-exactness is won or lost.
// ---------------------------------------------------------------------------

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <utility>

namespace torchstrap {

// Elements per (replica, chunk) task. Must be a multiple of Vectorized::size() so
// that the ragged `T % Vec::size()` tail lands only at the end of a replica's row
// -- ATen chunks by cache lines, which has the same property, and the tail is the
// one place its vectorized and scalar formulations disagree.
constexpr int64_t kChunk = 16384;

// Drive `fn(r, off, n)` over every live (replica, chunk) task.
//
// A frozen replica is skipped outright: nothing of its row is read and nothing is
// written, so it stays bit-identical -- the same guarantee the CUDA kernels get
// from their block-uniform early exit, and required rather than merely nice (a
// replica frozen at step 0 has `bias_correction2 == 0`, so an arithmetic gate
// would leak a NaN into its row).
//
// `mask` must point at `R` contiguous bools that outlive the call.
template <typename Fn>
inline void parallel_for_chunks(int64_t R, int64_t T, const bool* mask, Fn&& fn) {
  const int64_t n_chunks = (T + kChunk - 1) / kChunk;

  at::parallel_for(0, R * n_chunks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t r = idx / n_chunks;
      if (!mask[r]) {
        continue;
      }
      const int64_t off = (idx % n_chunks) * kChunk;
      fn(r, off, std::min(kChunk, T - off));
    }
  });
}

} // namespace torchstrap
