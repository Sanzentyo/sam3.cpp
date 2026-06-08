#pragma once

#include <cstddef>
#include <cstdint>

struct Sam3F32BatchedCustomResult {
    double mean_ms;
    double median_ms;
    double p95_ms;
    double min_ms;
    double max_ms;
};

struct Sam3GeluQuantDs4BenchResult {
    Sam3F32BatchedCustomResult separate;
    Sam3F32BatchedCustomResult fused;
    float max_abs_output_delta;
    int q8_diff_bytes;
};

enum class Sam3Bf16CublasLtCompute {
    fp32,
    fast_16bf,
};

struct Sam3Bf16CublasLtOptions {
    bool direct_bf16_dst;
    bool gelu_epilogue;
    Sam3Bf16CublasLtCompute compute;
};

[[nodiscard]] int sam3_f32_batched_custom_bench(const float* weights,
                                                const float* input,
                                                const float* bias,
                                                float* output,
                                                int64_t k,
                                                int64_t rows,
                                                int64_t cols,
                                                int64_t batches,
                                                int warmup,
                                                int iters,
                                                Sam3F32BatchedCustomResult* result);

[[nodiscard]] int sam3_f32_batched_cublaslt_bench(const float* weights,
                                                  const float* input,
                                                  const float* bias,
                                                  float* output,
                                                  int64_t k,
                                                  int64_t rows,
                                                  int64_t cols,
                                                  int64_t batches,
                                                  int warmup,
                                                  int iters,
                                                  bool fast_tf32,
                                                  Sam3F32BatchedCustomResult* result);

[[nodiscard]] int sam3_bf16_batched_cublaslt_bench(const float* weights,
                                                   const float* input,
                                                   const float* bias,
                                                   float* output,
                                                   int64_t k,
                                                   int64_t rows,
                                                   int64_t cols,
                                                   int64_t batches,
                                                   int warmup,
                                                   int iters,
                                                   Sam3Bf16CublasLtOptions options,
                                                   Sam3F32BatchedCustomResult* result);

[[nodiscard]] int sam3_gelu_quant_ds4_bench(const float* input,
                                            const float* bias,
                                            int64_t rows,
                                            int64_t cols,
                                            int warmup,
                                            int iters,
                                            Sam3GeluQuantDs4BenchResult* result);
