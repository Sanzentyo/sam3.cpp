#pragma once

#include <cstddef>
#include <cstdint>

struct sam3_cuda_bicubic_preprocess_cache {
    void* src_device = nullptr;
    size_t src_device_bytes = 0;
    void* horizontal_device = nullptr;
    size_t horizontal_device_bytes = 0;
    void* x_start_device = nullptr;
    void* x_count_device = nullptr;
    void* x_coeff_device = nullptr;
    void* y_start_device = nullptr;
    void* y_count_device = nullptr;
    void* y_coeff_device = nullptr;
    int x_src = 0;
    int x_dst = 0;
    int x_ksize = 0;
    int y_src = 0;
    int y_dst = 0;
    int y_ksize = 0;
};

bool sam3_cuda_preprocess_image_chw(const uint8_t* src,
                                    int src_w,
                                    int src_h,
                                    float* dst,
                                    int dst_size,
                                    const float mean[3],
                                    const float std_d[3]);

bool sam3_cuda_preprocess_image_chw_cached(const uint8_t* src,
                                           int src_w,
                                           int src_h,
                                           float* dst,
                                           int dst_size,
                                           const float mean[3],
                                           const float std_d[3],
                                           void** src_device_cache,
                                           size_t* src_device_cache_bytes);

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
                                                        sam3_cuda_bicubic_preprocess_cache* cache);

void sam3_cuda_preprocess_free_cache(void*& src_device_cache);

void sam3_cuda_bicubic_preprocess_free_cache(sam3_cuda_bicubic_preprocess_cache& cache);
