#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <cstdio>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define unimplemented(...) do { fprintf(stderr, "not implemented\n"); assert(false); __builtin_unreachable(); } while (false);

#define CUDA_SAFE_CALL(call) { \
    auto err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %d (%s) at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    } \
}

inline const char* cuBlasGetErrorString(cublasStatus_t err) {
    switch (err) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "UNKNOWN";
    }
}

#define CUBLAS_SAFE_CALL(call) { \
    auto err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error %s at %s:%d\n", cuBlasGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    } \
}

#define CHECK_CUDA_POINTER(ptr) if constexpr (DEBUG) { \
    cudaPointerAttributes attr; \
    CUDA_SAFE_CALL(cudaPointerGetAttributes(&attr, ptr)); \
    if (attr.type != cudaMemoryTypeDevice) { \
        fprintf(stderr, "Wrong CUDA pointer %s type: %d\n", #ptr, attr.type); \
        assert(false); \
    } \
}

template <typename T>
inline void showCudaArray(T *d_value, int m, int n) {
    auto temp = new std::remove_cv_t<T>[m * n]();
    CUDA_SAFE_CALL(cudaMemcpy(temp, d_value, sizeof(T) * m * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << temp[i * n + j];
            std::cout << ((j == n - 1) ? '\n' : ' ');
        }
    }
    delete temp;
}

template <typename T>
inline void showArray(T *value, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << value[i * n + j];
            std::cout << ((j == n - 1) ? '\n' : ' ');
        }
    }
}


#endif // UTILITY_H
