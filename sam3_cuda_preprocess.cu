#include "sam3_cuda_preprocess.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void sam3_preprocess_image_chw_kernel(const uint8_t* __restrict__ src,
                                                 int src_w,
                                                 int src_h,
                                                 float* __restrict__ dst,
                                                 int dst_size,
                                                 float mean0,
                                                 float mean1,
                                                 float mean2,
                                                 float std0,
                                                 float std1,
                                                 float std2) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int area = dst_size * dst_size;
    if (idx >= area) {
        return;
    }

    const int x = idx % dst_size;
    const int y = idx / dst_size;
    const int chw_base = idx;

    if (src_w == dst_size && src_h == dst_size) {
        const int hwc_base = ((y * src_w) + x) * 3;
        dst[chw_base] = ((static_cast<float>(src[hwc_base]) / 255.0f) - mean0) / std0;
        dst[area + chw_base] = ((static_cast<float>(src[hwc_base + 1]) / 255.0f) - mean1) / std1;
        dst[(2 * area) + chw_base] =
            ((static_cast<float>(src[hwc_base + 2]) / 255.0f) - mean2) / std2;
        return;
    }

    const double scale_x = static_cast<double>(src_w) / static_cast<double>(dst_size);
    const double scale_y = static_cast<double>(src_h) / static_cast<double>(dst_size);
    double fx = ((static_cast<double>(x) + 0.5) * scale_x) - 0.5;
    double fy = ((static_cast<double>(y) + 0.5) * scale_y) - 0.5;
    fx = fmax(fx, 0.0);
    fy = fmax(fy, 0.0);

    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const int x1 = (x0 < src_w - 1) ? x0 + 1 : x0;
    const int y1 = (y0 < src_h - 1) ? y0 + 1 : y0;
    const double wx = fx - static_cast<double>(x0);
    const double wy = fy - static_cast<double>(y0);
    const double wx0 = 1.0 - wx;
    const double wy0 = 1.0 - wy;

    const int p00 = ((y0 * src_w) + x0) * 3;
    const int p01 = ((y0 * src_w) + x1) * 3;
    const int p10 = ((y1 * src_w) + x0) * 3;
    const int p11 = ((y1 * src_w) + x1) * 3;

    const float means[3] = {mean0, mean1, mean2};
    const float stds[3] = {std0, std1, std2};
    for (int c = 0; c < 3; ++c) {
        const double resized = (wy0 * ((wx0 * static_cast<double>(src[p00 + c])) +
                                       (wx * static_cast<double>(src[p01 + c])))) +
                               (wy * ((wx0 * static_cast<double>(src[p10 + c])) +
                                      (wx * static_cast<double>(src[p11 + c]))));
        const int iv = min(max(static_cast<int>(llround(resized)), 0), 255);
        const float v = static_cast<float>(iv) / 255.0f;
        dst[(c * area) + chw_base] = (v - means[c]) / stds[c];
    }
}

}  // namespace

bool sam3_cuda_preprocess_image_chw(const uint8_t* src,
                                    int src_w,
                                    int src_h,
                                    float* dst,
                                    int dst_size,
                                    const float mean[3],
                                    const float std_d[3]) {
    void* cache = nullptr;
    size_t cache_bytes = 0;
    const bool ok = sam3_cuda_preprocess_image_chw_cached(
        src, src_w, src_h, dst, dst_size, mean, std_d, &cache, &cache_bytes);
    sam3_cuda_preprocess_free_cache(cache);
    return ok;
}

bool sam3_cuda_preprocess_image_chw_cached(const uint8_t* src,
                                           int src_w,
                                           int src_h,
                                           float* dst,
                                           int dst_size,
                                           const float mean[3],
                                           const float std_d[3],
                                           void** src_device_cache,
                                           size_t* src_device_cache_bytes) {
    if (src == nullptr || dst == nullptr || src_w <= 0 || src_h <= 0 || dst_size <= 0) {
        return false;
    }
    if (src_device_cache == nullptr || src_device_cache_bytes == nullptr) {
        return false;
    }

    const size_t src_bytes =
        static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * static_cast<size_t>(3);
    if (*src_device_cache_bytes < src_bytes) {
        if (*src_device_cache != nullptr) {
            cudaFree(*src_device_cache);
            *src_device_cache = nullptr;
            *src_device_cache_bytes = 0;
        }
        if (cudaMalloc(src_device_cache, src_bytes) != cudaSuccess) {
            return false;
        }
        *src_device_cache_bytes = src_bytes;
    }

    auto* src_device = static_cast<uint8_t*>(*src_device_cache);
    bool ok =
        cudaMemcpyAsync(src_device, src, src_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) ==
        cudaSuccess;
    if (ok) {
        constexpr int block_size = 256;
        const int area = dst_size * dst_size;
        const int grid_size = (area + block_size - 1) / block_size;
        sam3_preprocess_image_chw_kernel<<<grid_size, block_size, 0, cudaStreamPerThread>>>(
            src_device,
            src_w,
            src_h,
            dst,
            dst_size,
            mean[0],
            mean[1],
            mean[2],
            std_d[0],
            std_d[1],
            std_d[2]);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }

    return ok;
}

void sam3_cuda_preprocess_free_cache(void*& src_device_cache) {
    if (src_device_cache == nullptr) {
        return;
    }
    cudaFree(src_device_cache);
    src_device_cache = nullptr;
}
