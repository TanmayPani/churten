"""Ahead-of-time build of `torchstrap.kernels._C`.

One extension, both kernels: `kernels/csrc/stubs.cpp` declares the operator, and
`kernels/csrc/cpu/adam.cpp` plus (when a usable CUDA toolkit exists)
`kernels/csrc/cuda/adam.cu` register themselves on their dispatch keys,
exactly as ATen does for `_fused_adam_`. Importing `torchstrap.kernels._C` is what
runs those static initializers. Without a CUDA toolkit the extension is CPU-only and the op raises
from the dispatcher on a CUDA tensor, which is also what `torch._fused_adam_`
does on a device ATen has no kernel for.

Building needs torch *already installed*, because the extension links against
this exact torch's libraries and ABI. Hence `no-build-isolation-package` in
pyproject.toml: a fresh torch downloaded into an isolated build env would be a
different build (possibly a different CUDA) than the one at runtime.

    uv sync                                  # first install
    uv sync --reinstall-package torchstrap   # after editing anything in csrc/

Everything below the flag tables is toolchain *discovery*, not configuration:
which CUDA toolkit, which host compiler, which vector ISA. There is deliberately
no torchstrap-specific environment variable. `TORCH_CUDA_ARCH_LIST` is honoured
because it is torch's own build-time convention and `CUDAExtension` reads it
directly; unset, torch targets the GPUs it can see.
"""

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from shutil import which

from setuptools import setup

# setuptools requires source paths relative to setup.py, never absolute.
CSRC = Path("src") / "torchstrap" / "kernels" / "csrc"


# ---------------------------------------------------------------------------
# Compile flags. These are NUMERICS, not just speed -- see below.
# ---------------------------------------------------------------------------

# `-O2` (not -O3) and GCC's default `-ffp-contract=fast` together reproduce how
# ATen builds FusedAdamKernel.cpp, which is what makes csrc/cpu/adam.cpp
# bit-identical to `torch._fused_adam_` on CPU. Both halves are load-bearing and
# fail in OPPOSITE directions: at -O3 GCC fuses AdamW's `param * (1 - lr*wd)`
# into the final update where ATen at -O2 does not; with -ffp-contract=off plain
# Adam's `grad += param * wd` fails to fuse where ATen's does. Either way, 1 ulp.
_CXX_FLAGS = ["-O2", "-fno-math-errno"]

# `-DCPU_CAPABILITY_AVX2` selects ATen's real AVX2 `Vectorized<float>` instead of
# vec_base.h's array-of-scalars fallback, so `at::vec::fmadd` becomes
# `_mm256_fmadd_ps` -- the instruction ATen issues for the exp_avg lerp. It drags
# in `-mf16c`: torch's vec_half.h then reaches for the F16C intrinsic `_cvtsh_ss`
# and without the flag GCC refuses to inline an always_inline builtin.
#
# Gated on the build host actually having AVX2, because unlike the old JIT build
# this object can be installed once and outlive a CPU migration. `-march=native`
# is still avoided for the same reason.
_AVX2_FLAGS = ["-mavx2", "-mfma", "-mf16c", "-DCPU_CAPABILITY_AVX2"]

# `-O2` for the same contraction reason. The launch shape is ATen's own
# (kBlockSize = 512, kChunkSize = 65536) and deliberately not autotuned: the
# update is bandwidth-bound, so at the real workload size (R=100, T=265k on a
# 4070 Laptop) block sizes 128/256/512/1024 all land within 1% of each other.
#
# `--expt-relaxed-constexpr` is required because ATen's `pow_` (Pow.cuh) is a
# `__host__ __device__` function that calls std::pow. `--extended-lambda` is
# required by ATen's own headers -- MultiTensorApply.cuh drags in MemoryAccess.cuh
# and CUDALoops.cuh, which declare `__device__` lambdas -- and is what torch
# builds ATen with. The old JIT path got away without it only because
# `cpp_extension.load` puts torch's include tree behind `-isystem` while
# `CUDAExtension` uses `-I`; nothing in our kernel uses a device lambda, so this
# cannot move the arithmetic.
_NVCC_FLAGS = [
    "-O2",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-lineinfo",
]


def _host_has_avx2() -> bool:
    try:
        flags = set()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    except OSError:
        return False
    return {"avx2", "fma", "f16c"} <= flags


# ---------------------------------------------------------------------------
# CUDA toolchain discovery
# ---------------------------------------------------------------------------


def _nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [str(nvcc), "--version"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _pick_cuda_home(torch_cuda: str | None) -> str | None:
    """The installed CUDA toolkit closest to torch's own CUDA version.

    The **major** must match: torch's bundled ``libcudart.so.<major>`` is what is
    loaded at runtime, so crossing a major is not a compatibility question but a
    different library. Within that major an exact minor wins, then the newest at
    or below torch's (the supported direction, since cudart is backward but not
    forward compatible), and only then a newer minor -- which usually works,
    because the entry points nvcc emits for a plain kernel launch are long
    stable, but is the thing to suspect on an undefined-symbol import failure.
    """
    if torch_cuda is None:
        return None
    parts = [int(x) for x in torch_cuda.split(".")[:2]]
    want: tuple[int, int] = (parts[0], parts[1] if len(parts) > 1 else 0)

    roots = sorted(Path("/usr/local").glob("cuda-*")) + [Path("/usr/local/cuda")]
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    nvcc_on_path = which("nvcc")
    if nvcc_on_path:
        roots.append(Path(nvcc_on_path).resolve().parent.parent)

    candidates: dict[tuple[int, int], str] = {}
    for root in roots:
        nvcc = root / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        ver = _nvcc_version(nvcc)
        if ver is not None and ver[0] == want[0]:
            candidates.setdefault(ver, str(root.resolve()))

    if not candidates:
        return None
    if want in candidates:
        return candidates[want]
    at_or_below = [v for v in candidates if v <= want]
    if at_or_below:
        return candidates[max(at_or_below)]
    return candidates[min(candidates)]


def _gcc_ceiling(cuda_home: str) -> int | None:
    """Max __GNUC__ this toolkit accepts, read from its own crt/host_config.h."""
    header = Path(cuda_home) / "include" / "crt" / "host_config.h"
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"__GNUC__\s*>\s*(\d+)", text)
    return int(m.group(1)) if m else None


@lru_cache(maxsize=16)
def _gnuc_major(exe: str) -> int | None:
    try:
        out = subprocess.run(
            [exe, "-dumpversion"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"(\d+)", out.strip())
    return int(m.group(1)) if m else None


def _pick_ccbin(cuda_home: str) -> str | None:
    """Newest installed g++ this toolkit accepts as its host compiler.

    None when the default `c++` is already within the ceiling, which leaves nvcc
    on its own default.

    ``/usr/bin`` is searched before ``PATH`` deliberately. A Homebrew/linuxbrew
    GCC ships libstdc++ headers built against a different glibc, and feeding one
    to nvcc fails deep inside ``<ext/concurrence.h>`` ("too many initializer
    values") rather than anywhere near our source. The distro compiler that
    matches the running glibc is always the right answer here.
    """
    ceiling = _gcc_ceiling(cuda_home)
    if ceiling is None:
        return None

    default = _gnuc_major("c++")
    if default is not None and default <= ceiling:
        return None

    for major in range(ceiling, 3, -1):
        for name in (f"g++-{major}", f"g++{major}"):
            for exe in (f"/usr/bin/{name}", which(name)):
                if (
                    exe is not None
                    and Path(exe).is_file()
                    and (_gnuc_major(exe) or 0) <= ceiling
                ):
                    return exe
    return None


# ---------------------------------------------------------------------------
# Assemble the extension
# ---------------------------------------------------------------------------

import torch  # noqa: E402 - after the discovery helpers, before cpp_extension

cuda_home = _pick_cuda_home(torch.version.cuda)
if cuda_home is not None:
    # cpp_extension resolves CUDA_HOME at import time, so this has to land first.
    os.environ["CUDA_HOME"] = cuda_home

from torch.utils.cpp_extension import (  # noqa: E402
    BuildExtension,
    CUDAExtension,
    CppExtension,
)

with_cuda = cuda_home is not None and torch.version.hip is None
ccbin = _pick_ccbin(cuda_home) if with_cuda and cuda_home else None
nvcc_ver = _nvcc_version(Path(cuda_home) / "bin" / "nvcc") if cuda_home else None

sources = [
    (CSRC / "stubs.cpp").as_posix(),
    (CSRC / "cpu" / "adam.cpp").as_posix(),
    (CSRC / "cpu" / "sgd.cpp").as_posix(),
    (CSRC / "cpu" / "adagrad.cpp").as_posix(),
]
cxx_flags = list(_CXX_FLAGS) + (_AVX2_FLAGS if _host_has_avx2() else [])
nvcc_flags = list(_NVCC_FLAGS)

if with_cuda:
    sources.append((CSRC / "cuda" / "adam.cu").as_posix())
    sources.append((CSRC / "cuda" / "sgd.cu").as_posix())
    sources.append((CSRC / "cuda" / "adagrad.cu").as_posix())
    if ccbin is not None:
        nvcc_flags += ["-ccbin", ccbin]

factory = CUDAExtension if with_cuda else CppExtension
ext = factory(
    name="torchstrap.kernels._C",
    sources=sources,
    extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
)

print(
    f"torchstrap: building _C with_cuda={with_cuda} cuda_home={cuda_home} "
    f"nvcc={nvcc_ver} ccbin={ccbin} avx2={_host_has_avx2()}"
)

setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
