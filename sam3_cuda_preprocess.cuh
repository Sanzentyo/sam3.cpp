#pragma once

#include <cstddef>
#include <cstdint>

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

void sam3_cuda_preprocess_free_cache(void*& src_device_cache);
