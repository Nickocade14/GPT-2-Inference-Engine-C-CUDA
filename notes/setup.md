# Environment Setup

## Hardware
- GPU: NVIDIA RTX 4060 **Laptop** GPU (Ada Lovelace, AD107)
  - compute capability 8.9 (`sm_89`), 24 SMs, 8 GB VRAM
- OS: Windows 11, WDDM driver model

## Installed
- CUDA Toolkit 13.3 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`)
- Driver 610.88 (CUDA UMD 13.3 — matches the toolkit)
- MSVC 19.50 via VS 2026 Build Tools (`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`)
- VS Code + C/C++ extension (Microsoft)

## IMPORTANT: run in every new terminal
`nvcc` needs `cl.exe`, and MSVC is not on the global PATH.
Before compiling, run:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

The `-Arch amd64 -HostArch amd64` flags are required: without them the script
loads x86, and CUDA ships 64-bit libraries only.

Verify: `cl` must print "for x64".

## Build

```powershell
nvcc -arch=sm_89 src\file.cu -o build\file.exe
```

---

## Gotcha #1: always pass `-arch=sm_89`

`nvcc` 13.3 defaults to `arch = sm_75` (Turing). This GPU is `sm_89`.
Confirmed with `cuobjdump -res-usage build\file.exe`.

Without `-arch`, the driver JIT-compiles the embedded `compute_75` PTX onto Ada
at load time. **This produces correct results** — verified 2026-08-04, the test
kernel returned the expected values — so there is no visible symptom. What you
get is Turing-targeted codegen translated onto Ada, plus JIT latency on every
launch. Since this project is about comparing kernel versions, that means
measuring against a baseline that is not the real hardware.

`-arch=sm_89` emits native SASS and skips the JIT entirely.

## Gotcha #2: `cudaDeviceSynchronize()` does not catch launch failures

These are two different errors and both must be checked:

- `cudaGetLastError()` — the **launch** was rejected (bad arch, bad grid/block
  dims, too much shared memory). Nothing was ever enqueued.
- `cudaDeviceSynchronize()` — the kernel ran and **execution** failed
  (illegal address, etc).

When a launch is rejected, nothing is queued, so there is nothing to sync on
and `cudaDeviceSynchronize()` correctly returns `no error`. Checking only the
second one hides launch failures completely.

```cpp
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch
CUDA_CHECK(cudaDeviceSynchronize());   // execution
```

## Troubleshooting
- "The term X is not recognized" → PATH issue.
  Close and reopen the terminal (or all of VS Code) after installing anything.
- `nvidia-smi` ships with the driver, `nvcc` with the toolkit. Different things.
