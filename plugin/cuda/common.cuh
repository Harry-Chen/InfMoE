#pragma once

#ifndef COMMON_CUH
#define COMMON_CUH

static const int WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize,
                                           unsigned int mask = 0xffffffff) {
    return __shfl_xor_sync(mask, value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff) {
    return __shfl_sync(mask, value, srcLane, width);
}

template <typename T>
__device__ __host__ __forceinline__ T ceiling(T m, T n) {
    return (m + (n - 1)) / n;
}

template <typename T>
__device__ __host__ __forceinline__ T sigmoid(T x) {
    return 1 / (1 + exp(-x));
}

#endif // COMMON_CUH
