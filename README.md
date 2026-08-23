# GPT-2 Inference Engine in C++/CUDA
A C++/CUDA inference engine that runs GPT-2 small on an RTX 4060. I write the kernels by hand, starting with matmul, and optimize them across successive versions, benchmarking each one against the last. PyTorch is used exclusively to extract the weights of the LLM and generate the results to compare if my engine is generating the correct result.

## Main Purpose
Learning GPU programming from scratch: how work is distributed across thousands of threads, and why one version beats another.

## Progress (Weeks 1–5)

- Generated a reference inference output for GPT-2 using PyTorch, which serves as the source of truth for validating the engine.
- Inspected the model's tensors, including names, shapes, strides, parameter counts, and total bytes.
- Ran a memory layout experiment by creating and transposing a tensor, then checking storage sharing and contiguity.
- Implemented matrix multiplication from scratch in C++ using flattened 1D arrays and row-major indexing, as the CPU reference for validating the CUDA kernel. Output verified against a manual calculation.

Not started yet: the CUDA matmul kernel, the binary weight serializer, and the forward pass.

## Folder Structure

- src/ — C++/CUDA engine
- tools/ — Weight extraction and reference generation
- notes/ — Environment setup and documented findings
- reports/ — Inspection script outputs
- tests/ — Tests, currently empty

## Build

Requires the CUDA Toolkit and MSVC. Run once per new terminal:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

Then compile and run:

```powershell
mkdir build
nvcc -arch=sm_89 src\matmul_cpu.cu -o build\matmul_cpu.exe
.\build\matmul_cpu.exe
```

Developed and tested on an RTX 4060 Laptop (`sm_89`). Other GPUs require a
different `-arch` value.

See [notes/setup.md](notes/setup.md) for the full environment setup.

## Roadmap

1. **Weeks 1–5** — Hand-written CUDA matmul kernel, validated against the CPU reference.
2. **Weeks 6–9** — Extract GPT-2 weights into a custom binary format, load them into VRAM, verify shapes.
3. **Weeks 10–16** — Full forward pass, layer by layer, validated against PyTorch.
4. **Week 17** — Generation loop.
5. **Months 5–12** — Kernel optimization.