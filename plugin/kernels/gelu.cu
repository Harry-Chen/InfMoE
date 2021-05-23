#include <thrust/device_vector.h>
#include <thrust/transform.h>

template <typename T>
struct gelu_dot {
    constexpr static T A = 0.5;
    constexpr static T B = 0.7978845608028654;    // sqrt(2.0/M_PI)
    constexpr static T C = 0.035677408136300125;  // 0.044715 * sqrt(2.0/M_PI)
    //  gelu(A) * b
    __host__ __device__ inline T operator()(const T &a, const T &b) const {
        const T cdf = A * (1 + tanh(a * (C * a * a + B)));
        return a * cdf * b;
    }
};

template <typename T>
void fused_gelu_dot_kernel(T *A, T *B, size_t len, cudaStream_t stream) {
    auto dev_A = thrust::device_pointer_cast(A);
    auto dev_B = thrust::device_pointer_cast(B);
    thrust::transform(thrust::cuda::par.on(stream), dev_A, dev_A + len, dev_B, dev_B, gelu_dot<T>{});
}

template void fused_gelu_dot_kernel(float *A, float *B, size_t len, cudaStream_t stream);
