#include "sam3_cuda_preprocess.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int BICUBIC_PRECISION_BITS = 32 - 8 - 2;

__device__ uint8_t sam3_bicubic_clip8_device(int32_t value) {
    const int scaled = value >> BICUBIC_PRECISION_BITS;
    return static_cast<uint8_t>(min(max(scaled, 0), 255));
}

__device__ float sam3_fp16_storage_normalize_device(float pixel01, float mean, float std_d) {
    const float stored = __half2float(__float2half_rn(pixel01));
    const float mean_h = __half2float(__float2half_rn(mean));
    const float std_h = __half2float(__float2half_rn(std_d));
    const float centered = __half2float(__float2half_rn(stored - mean_h));
    return __half2float(__float2half_rn(centered / std_h));
}

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
        dst[chw_base] = sam3_fp16_storage_normalize_device(
            static_cast<float>(src[hwc_base]) / 255.0f, mean0, std0);
        dst[area + chw_base] = sam3_fp16_storage_normalize_device(
            static_cast<float>(src[hwc_base + 1]) / 255.0f, mean1, std1);
        dst[(2 * area) + chw_base] = sam3_fp16_storage_normalize_device(
            static_cast<float>(src[hwc_base + 2]) / 255.0f, mean2, std2);
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
        dst[(c * area) + chw_base] =
            sam3_fp16_storage_normalize_device(v, means[c], stds[c]);
    }
}

__global__ void sam3_bicubic_direct_norm_fp16_storage_kernel(const uint8_t* __restrict__ src,
                                                             int src_w,
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
    const int hwc_base = ((y * src_w) + x) * 3;
    dst[idx] =
        sam3_fp16_storage_normalize_device(static_cast<float>(src[hwc_base]) / 255.0f, mean0, std0);
    dst[area + idx] = sam3_fp16_storage_normalize_device(
        static_cast<float>(src[hwc_base + 1]) / 255.0f, mean1, std1);
    dst[(2 * area) + idx] = sam3_fp16_storage_normalize_device(
        static_cast<float>(src[hwc_base + 2]) / 255.0f, mean2, std2);
}

__global__ void sam3_bicubic_horizontal_u8_kernel(const uint8_t* __restrict__ src,
                                                  int src_w,
                                                  int src_h,
                                                  uint8_t* __restrict__ horizontal,
                                                  int dst_size,
                                                  const int* __restrict__ x_start,
                                                  const int* __restrict__ x_count,
                                                  const int32_t* __restrict__ x_coeff,
                                                  int x_ksize) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int total = src_h * dst_size * 3;
    if (idx >= total) {
        return;
    }

    const int c = idx % 3;
    const int x = (idx / 3) % dst_size;
    const int y = idx / (3 * dst_size);
    const int xmin = x_start[x];
    const int count = x_count[x];
    const int32_t* coeff = x_coeff + (x * x_ksize);

    int32_t acc = 1 << (BICUBIC_PRECISION_BITS - 1);
    for (int k = 0; k < count; ++k) {
        acc += static_cast<int32_t>(src[((y * src_w) + xmin + k) * 3 + c]) * coeff[k];
    }
    horizontal[idx] = sam3_bicubic_clip8_device(acc);
}

__global__ void sam3_bicubic_vertical_norm_fp16_storage_kernel(
    const uint8_t* __restrict__ horizontal,
    float* __restrict__ dst,
    int dst_size,
    const int* __restrict__ y_start,
    const int* __restrict__ y_count,
    const int32_t* __restrict__ y_coeff,
    int y_ksize,
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
    const int ymin = y_start[y];
    const int count = y_count[y];
    const int32_t* coeff = y_coeff + (y * y_ksize);
    const float means[3] = {mean0, mean1, mean2};
    const float stds[3] = {std0, std1, std2};

    for (int c = 0; c < 3; ++c) {
        int32_t acc = 1 << (BICUBIC_PRECISION_BITS - 1);
        for (int k = 0; k < count; ++k) {
            acc += static_cast<int32_t>(horizontal[((ymin + k) * dst_size + x) * 3 + c]) * coeff[k];
        }
        const float pixel01 = static_cast<float>(sam3_bicubic_clip8_device(acc)) / 255.0f;
        dst[(c * area) + idx] = sam3_fp16_storage_normalize_device(pixel01, means[c], stds[c]);
    }
}

bool sam3_cuda_ensure_device_buffer(void** ptr, size_t* capacity, size_t required) {
    if (ptr == nullptr || capacity == nullptr || required == 0) {
        return false;
    }
    if (*capacity >= required && *ptr != nullptr) {
        return true;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    if (cudaMalloc(ptr, required) != cudaSuccess) {
        return false;
    }
    *capacity = required;
    return true;
}

bool sam3_cuda_replace_raw_buffer(void** ptr, size_t required) {
    if (ptr == nullptr || required == 0) {
        return false;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
    }
    return cudaMalloc(ptr, required) == cudaSuccess;
}

bool sam3_cuda_upload_bicubic_axis(const int* start,
                                   const int* count,
                                   const int32_t* coeff,
                                   int src,
                                   int dst,
                                   int ksize,
                                   void*& start_device,
                                   void*& count_device,
                                   void*& coeff_device,
                                   int& cached_src,
                                   int& cached_dst,
                                   int& cached_ksize) {
    if (start == nullptr || count == nullptr || coeff == nullptr || src <= 0 || dst <= 0 ||
        ksize <= 0) {
        return false;
    }
    if (cached_src == src && cached_dst == dst && cached_ksize == ksize &&
        start_device != nullptr && count_device != nullptr && coeff_device != nullptr) {
        return true;
    }

    const size_t vector_bytes = static_cast<size_t>(dst) * sizeof(int);
    const size_t coeff_bytes =
        static_cast<size_t>(dst) * static_cast<size_t>(ksize) * sizeof(int32_t);
    if (!sam3_cuda_replace_raw_buffer(&start_device, vector_bytes) ||
        !sam3_cuda_replace_raw_buffer(&count_device, vector_bytes) ||
        !sam3_cuda_replace_raw_buffer(&coeff_device, coeff_bytes)) {
        return false;
    }

    bool ok = cudaMemcpyAsync(
                  start_device, start, vector_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) ==
              cudaSuccess;
    ok =
        ok && cudaMemcpyAsync(
                  count_device, count, vector_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) ==
                  cudaSuccess;
    ok = ok && cudaMemcpyAsync(
                   coeff_device, coeff, coeff_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) ==
                   cudaSuccess;
    if (!ok) {
        return false;
    }

    cached_src = src;
    cached_dst = dst;
    cached_ksize = ksize;
    return true;
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
    if (!sam3_cuda_ensure_device_buffer(src_device_cache, src_device_cache_bytes, src_bytes)) {
        return false;
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

bool sam3_cuda_preprocess_image_chw_bicubic_fp16_cached(const uint8_t* src,
                                                        int src_w,
                                                        int src_h,
                                                        float* dst,
                                                        int dst_size,
                                                        const float mean[3],
                                                        const float std_d[3],
                                                        const int* x_start,
                                                        const int* x_count,
                                                        const int32_t* x_coeff,
                                                        int x_ksize,
                                                        const int* y_start,
                                                        const int* y_count,
                                                        const int32_t* y_coeff,
                                                        int y_ksize,
                                                        sam3_cuda_bicubic_preprocess_cache* cache) {
    if (src == nullptr || dst == nullptr || cache == nullptr || src_w <= 0 || src_h <= 0 ||
        dst_size <= 0) {
        return false;
    }

    const size_t src_bytes =
        static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * static_cast<size_t>(3);
    if (!sam3_cuda_ensure_device_buffer(&cache->src_device, &cache->src_device_bytes, src_bytes)) {
        return false;
    }

    auto* src_device = static_cast<uint8_t*>(cache->src_device);
    bool ok =
        cudaMemcpyAsync(src_device, src, src_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread) ==
        cudaSuccess;
    if (!ok) {
        return false;
    }

    constexpr int block_size = 256;
    const int area = dst_size * dst_size;
    if (src_w == dst_size && src_h == dst_size) {
        const int grid_size = (area + block_size - 1) / block_size;
        sam3_bicubic_direct_norm_fp16_storage_kernel<<<grid_size,
                                                       block_size,
                                                       0,
                                                       cudaStreamPerThread>>>(src_device,
                                                                              src_w,
                                                                              dst,
                                                                              dst_size,
                                                                              mean[0],
                                                                              mean[1],
                                                                              mean[2],
                                                                              std_d[0],
                                                                              std_d[1],
                                                                              std_d[2]);
        return cudaGetLastError() == cudaSuccess &&
               cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }

    if (!sam3_cuda_upload_bicubic_axis(x_start,
                                       x_count,
                                       x_coeff,
                                       src_w,
                                       dst_size,
                                       x_ksize,
                                       cache->x_start_device,
                                       cache->x_count_device,
                                       cache->x_coeff_device,
                                       cache->x_src,
                                       cache->x_dst,
                                       cache->x_ksize)) {
        return false;
    }
    if (!sam3_cuda_upload_bicubic_axis(y_start,
                                       y_count,
                                       y_coeff,
                                       src_h,
                                       dst_size,
                                       y_ksize,
                                       cache->y_start_device,
                                       cache->y_count_device,
                                       cache->y_coeff_device,
                                       cache->y_src,
                                       cache->y_dst,
                                       cache->y_ksize)) {
        return false;
    }

    const size_t horizontal_bytes =
        static_cast<size_t>(src_h) * static_cast<size_t>(dst_size) * static_cast<size_t>(3);
    if (!sam3_cuda_ensure_device_buffer(
            &cache->horizontal_device, &cache->horizontal_device_bytes, horizontal_bytes)) {
        return false;
    }

    const int horizontal_total = src_h * dst_size * 3;
    const int horizontal_grid = (horizontal_total + block_size - 1) / block_size;
    sam3_bicubic_horizontal_u8_kernel<<<horizontal_grid, block_size, 0, cudaStreamPerThread>>>(
        src_device,
        src_w,
        src_h,
        static_cast<uint8_t*>(cache->horizontal_device),
        dst_size,
        static_cast<const int*>(cache->x_start_device),
        static_cast<const int*>(cache->x_count_device),
        static_cast<const int32_t*>(cache->x_coeff_device),
        x_ksize);
    ok = cudaGetLastError() == cudaSuccess;

    const int vertical_grid = (area + block_size - 1) / block_size;
    if (ok) {
        sam3_bicubic_vertical_norm_fp16_storage_kernel<<<vertical_grid,
                                                         block_size,
                                                         0,
                                                         cudaStreamPerThread>>>(
            static_cast<const uint8_t*>(cache->horizontal_device),
            dst,
            dst_size,
            static_cast<const int*>(cache->y_start_device),
            static_cast<const int*>(cache->y_count_device),
            static_cast<const int32_t*>(cache->y_coeff_device),
            y_ksize,
            mean[0],
            mean[1],
            mean[2],
            std_d[0],
            std_d[1],
            std_d[2]);
        ok = cudaGetLastError() == cudaSuccess;
    }

    return ok && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
}

void sam3_cuda_preprocess_free_cache(void*& src_device_cache) {
    if (src_device_cache == nullptr) {
        return;
    }
    cudaFree(src_device_cache);
    src_device_cache = nullptr;
}

void sam3_cuda_bicubic_preprocess_free_cache(sam3_cuda_bicubic_preprocess_cache& cache) {
    if (cache.src_device != nullptr) {
        cudaFree(cache.src_device);
    }
    if (cache.horizontal_device != nullptr) {
        cudaFree(cache.horizontal_device);
    }
    if (cache.x_start_device != nullptr) {
        cudaFree(cache.x_start_device);
    }
    if (cache.x_count_device != nullptr) {
        cudaFree(cache.x_count_device);
    }
    if (cache.x_coeff_device != nullptr) {
        cudaFree(cache.x_coeff_device);
    }
    if (cache.y_start_device != nullptr) {
        cudaFree(cache.y_start_device);
    }
    if (cache.y_count_device != nullptr) {
        cudaFree(cache.y_count_device);
    }
    if (cache.y_coeff_device != nullptr) {
        cudaFree(cache.y_coeff_device);
    }
    cache = {};
}
