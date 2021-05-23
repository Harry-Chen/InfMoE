#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas.h>

#define unimplemented(...) do { fprintf(stderr, "not implemented\n"); assert(false); __builtin_unreachable(); } while (false);

#define CUDA_SAFE_CALL(call) { \
    auto err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
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



#endif // UTILITY_H
