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

enum class Sam3F16CublasLtCompute {
    fp32,
    fast_16f,
    compute_16f,
};

struct Sam3F16CublasLtOptions {
    Sam3F16CublasLtCompute compute;
    int algo_index;
    size_t workspace_bytes;
};

struct Sam3Bf16CublasLtOptions {
    bool direct_bf16_dst;
    bool c_f32_for_direct_dst;
    bool gelu_epilogue;
    Sam3Bf16CublasLtCompute compute;
    int algo_index;
    size_t workspace_bytes;
};

struct Sam3Bf16MlpChainResult {
    Sam3F32BatchedCustomResult chain;
    double output_checksum;
};

struct Sam3F16MlpChainOptions {
    Sam3F16CublasLtCompute compute;
    int fc1_algo_index;
    int fc2_algo_index;
    size_t workspace_bytes;
};

struct Sam3F16MlpChainResult {
    Sam3F32BatchedCustomResult chain;
    double output_checksum;
};

struct Sam3Bf16GeluCastResult {
    Sam3F32BatchedCustomResult gelu_cast;
    double output_checksum;
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

[[nodiscard]] int sam3_f16_batched_cublaslt_bench(const float* weights,
                                                  const float* input,
                                                  const float* bias,
                                                  float* output,
                                                  int64_t k,
                                                  int64_t rows,
                                                  int64_t cols,
                                                  int64_t batches,
                                                  int warmup,
                                                  int iters,
                                                  Sam3F16CublasLtOptions options,
                                                  Sam3F32BatchedCustomResult* result);

[[nodiscard]] int sam3_bf16_wmma_bench(const float* weights,
                                       const float* input,
                                       const float* bias,
                                       float* output,
                                       int64_t k,
                                       int64_t rows,
                                       int64_t cols,
                                       int64_t batches,
                                       int warmup,
                                       int iters,
                                       bool gelu_bf16_output,
                                       Sam3F32BatchedCustomResult* result);

[[nodiscard]] int sam3_bf16_mlp_chain_cublaslt_bench(const float* fc1_weights,
                                                     const float* fc2_weights,
                                                     const float* input,
                                                     const float* fc1_bias,
                                                     const float* fc2_bias,
                                                     float* output,
                                                     int64_t input_dim,
                                                     int64_t hidden_dim,
                                                     int64_t output_dim,
                                                     int64_t cols,
                                                     int64_t batches,
                                                     int warmup,
                                                     int iters,
                                                     Sam3Bf16CublasLtCompute compute,
                                                     size_t workspace_bytes,
                                                     bool direct_fc1_bf16_no_bias,
                                                     Sam3Bf16MlpChainResult* result);

[[nodiscard]] int sam3_f16_mlp_chain_cublaslt_bench(const float* fc1_weights,
                                                    const float* fc2_weights,
                                                    const float* input,
                                                    const float* fc1_bias,
                                                    const float* fc2_bias,
                                                    float* output,
                                                    int64_t input_dim,
                                                    int64_t hidden_dim,
                                                    int64_t output_dim,
                                                    int64_t cols,
                                                    int64_t batches,
                                                    int warmup,
                                                    int iters,
                                                    Sam3F16MlpChainOptions options,
                                                    Sam3F16MlpChainResult* result);

[[nodiscard]] int sam3_bf16_gelu_cast_bench(const float* input,
                                            float* output,
                                            int64_t rows,
                                            int64_t cols,
                                            int64_t batches,
                                            int warmup,
                                            int iters,
                                            Sam3Bf16GeluCastResult* result);

[[nodiscard]] int sam3_gelu_quant_ds4_bench(const float* input,
                                            const float* bias,
                                            int64_t rows,
                                            int64_t cols,
                                            int warmup,
                                            int iters,
                                            Sam3GeluQuantDs4BenchResult* result);
