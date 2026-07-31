# GPT-2-Inference-Engine-C-CUDA-
A C++/CUDA inference engine that runs GPT-2 small on an RTX 4060. I write the kernels by hand — starting with matmul — and optimize them across successive versions, benchmarking each one against the last. The goal is learning GPU programming from scratch: how work is distributed across thousands of threads, and why one version beats another.
