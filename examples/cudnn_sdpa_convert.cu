#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void convert_f32_to_bf16_kernel(const float* src, void* dst, size_t count) {
    auto* out = static_cast<__nv_bfloat16*>(dst);
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = __float2bfloat16(src[i]);
    }
}

__global__ void convert_f32_to_fp16_kernel(const float* src, void* dst, size_t count) {
    auto* out = static_cast<half*>(dst);
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = __float2half(src[i]);
    }
}

}  // namespace

extern "C" cudaError_t sam3_cudnn_sdpa_convert_f32_to_dtype(
    const float* src, void* dst, size_t count, int dtype_id) {
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    switch (dtype_id) {
        case 0:
            convert_f32_to_bf16_kernel<<<blocks, threads>>>(src, dst, count);
            break;
        case 1:
            convert_f32_to_fp16_kernel<<<blocks, threads>>>(src, dst, count);
            break;
        case 2:
            return cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice);
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
