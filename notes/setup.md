# Environment Setup

## Installation Paths and Toolchain

* GPU: NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace, `sm_89`, 8188 MiB VRAM)
* Driver 610.88 (WDDM), CUDA UMD 13.3
* CUDA Toolkit → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\`
* MSVC 19.50 / VS 2026 Build Tools →
  `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\`
* VS Code + Microsoft C/C++ extension
* MSYS2 is used for Unix-style shell utilities only. It is **not part of the
  CUDA compilation toolchain** — `nvcc` on Windows delegates host code to MSVC
  `cl.exe`, and the MSYS2 `g++` cannot serve as host compiler.

`19.50` is the version of `cl.exe`, not of Visual Studio. That is the number
that matters: each CUDA release declares a supported range of host compiler
versions and `nvcc` checks it at startup.

## IMPORTANT: Run in Every New Terminal

`nvcc` needs `cl.exe`, and MSVC is not on the global `PATH`.

Before compiling, run:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

The `-Arch amd64 -HostArch amd64` flags are required: without them the script
loads the x86 environment, and CUDA ships 64-bit libraries only.

Verify:

```powershell
cl
```

`cl` must report **"for x64"**. It prints version and architecture on the same
line, which covers both things this file asks you to check.

## Build

```powershell
nvcc -arch=sm_89 src\file.cu -o build\file.exe
```
Run it: 

```powershell
.\build\file.exe
```

Verify what actually landed in the binary:

```powershell
cuobjdump -lelf build\file.exe   # native SASS code objects
cuobjdump -lptx build\file.exe   # embedded PTX
```

`-lelf` should list an `sm_89` entry. `cuobjdump -res-usage` also reports the
architecture and is useful for register/shared-memory counts.

---

## Gotcha #1: Always Pass `-arch=sm_89`

`nvcc` 13.3 defaults to `sm_75` (Turing). This GPU is `sm_89` (Ada). Confirmed
with `cuobjdump -res-usage`.

**There is no visible symptom.** Verified 2026-08-04 on driver 610.88: a test
kernel built without `-arch` ran and returned the expected values. Correctness
does not prove the intended architecture was compiled — check the binary, not
the output.

What you get without `-arch` are two separate problems, and they do not have
the same weight:

**1. Codegen — not fixable at runtime.**
The embedded PTX was generated against the *virtual* architecture `compute_75`.
It therefore never had access to instructions, intrinsics, or assumptions
introduced after Turing. When the driver JIT-compiles that PTX onto Ada it
*translates* what is there; it does not regenerate it from the original source.
The result is Turing-targeted codegen running on Ada hardware. Nothing at
runtime recovers the missing information.

**2. Load-time JIT latency — amortizable.**
The JIT cost is paid when the module is loaded into the context, not on every
kernel launch. It is also cached by the driver on disk, so it is usually not
paid again across runs of the same binary. A warmup iteration hides it.

Problem 2 is an annoyance. **Problem 1 is the reason `-arch=sm_89` is
mandatory here**: this project exists to compare kernel versions against each
other, and a warmup makes the latency disappear while leaving you benchmarking
Turing-targeted code. The baseline would not be the real hardware.

`-arch=sm_89` emits native SASS for the actual GPU and skips the JIT entirely.

When recording benchmark results, record the driver version. "Worked on
2026-08-04" is historical context; the driver and toolkit versions are the
reproducible baseline — this GPU's behavior around JIT already changed once
across a driver update.

---

## Gotcha #2: `cudaDeviceSynchronize()` Does Not Catch Launch Failures

These are two different failure modes and both must be checked:

* `cudaGetLastError()` — the **launch** was rejected (bad architecture, invalid
  grid/block dimensions, too much shared memory). Nothing was ever enqueued.
* `cudaDeviceSynchronize()` — the kernel ran and **execution** failed (illegal
  memory access, etc).

The trap: when a launch is rejected, nothing is queued, so there is nothing to
synchronize on and `cudaDeviceSynchronize()` correctly returns `no error`.
Checking only the second call hides launch failures completely — not "makes
them harder to distinguish", hides them.

```cpp
#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

kernel<<<grid, block>>>(...);

CUDA_CHECK(cudaGetLastError());        // launch
CUDA_CHECK(cudaDeviceSynchronize());   // execution
```

---

## Troubleshooting

* `"The term X is not recognized"` → `PATH` or environment initialization.
  Close and reopen the terminal (or all of VS Code) after installing or
  modifying toolchain components.
* If `nvcc` cannot find `cl.exe`, run `Launch-VsDevShell.ps1` again and confirm
  `cl` reports x64.
* `nvidia-smi` ships with the **driver**; `nvcc` ships with the **Toolkit**.
  Different components, and the CUDA versions they report do not mean the same
  thing.
* If a binary runs correctly but timings look suspicious, inspect it with
  `cuobjdump` rather than assuming correct output proves the intended
  architecture was compiled. See Gotcha #1.
