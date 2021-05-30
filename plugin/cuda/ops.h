#pragma once

#ifndef OPS_H
#define OPS_H

#include <cstdint>
#include <cuda_runtime.h>

template <typename T, typename U>
void layernorm_gpu(T* __restrict__ output, const T* __restrict__ input,
                      int n1,  // batch_size * seq_length
                      int n2,  // embedding_size (or d_model)
                      double epsilon, // default to 1e-6
                      const T* gamma,  // weight, can be NULL
                      const T* beta,   // bias, can be NULL
                      const uint64_t maxGridY, // obtained from cudaGetDeviceProperties
                      cudaStream_t stream);

// B = gelu(A) . B
template <typename T>
void fused_gelu_dot_gpu(T* A, T* B, size_t len, cudaStream_t stream);

#endif  // OPS_H
