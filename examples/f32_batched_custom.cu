#include "f32_batched_custom.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr int kTile = 16;
constexpr int kQk8_1 = 32;
constexpr int kQuantBlockSizeMmq = 128;

namespace wmma = nvcuda::wmma;

struct BlockQ8_1MmqDs4 {
    half2 ds4[4];
    int8_t qs[4 * kQk8_1];
};

static_assert(sizeof(BlockQ8_1MmqDs4) == 4 * kQk8_1 + 4 * sizeof(half2));

__device__ __forceinline__ float gelu_tanh_f32(float x) {
    constexpr float kGeluCoefA = 0.044715f;
    constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * x * (1.0f + kGeluCoefA * x * x)));
}

__global__ void f32_batched_matmul_bias_kernel(const float* __restrict__ weights,
                                               const float* __restrict__ input,
                                               const float* __restrict__ bias,
                                               float* __restrict__ output,
                                               int k,
                                               int rows,
                                               int ncols) {
    __shared__ float w_tile[kTile][kTile];
    __shared__ float x_tile[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.0f;

    for (int kk0 = 0; kk0 < k; kk0 += kTile) {
        const int wk = kk0 + threadIdx.x;
        const int xk = kk0 + threadIdx.y;
        w_tile[threadIdx.y][threadIdx.x] = row < rows && wk < k ? weights[row * k + wk] : 0.0f;
        x_tile[threadIdx.y][threadIdx.x] = col < ncols && xk < k ? input[col * k + xk] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            acc += w_tile[threadIdx.y][kk] * x_tile[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && col < ncols) {
        output[col * rows + row] = acc + (bias != nullptr ? bias[row] : 0.0f);
    }
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ src,
                                   nv_bfloat16* __restrict__ dst,
                                   int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ src,
                                  half* __restrict__ dst,
                                  int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void bf16_to_f32_kernel(const nv_bfloat16* __restrict__ src,
                                   float* __restrict__ dst,
                                   int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __bfloat162float(src[i]);
    }
}

__global__ void gelu_erf_f32_to_bf16_kernel(const float* __restrict__ src,
                                            nv_bfloat16* __restrict__ dst,
                                            int64_t total) {
    constexpr float kSqrt1Over2 = 0.70710678118654752440084436210485f;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        const float x = src[i];
        const float y = 0.5f * x * (1.0f + erff(x * kSqrt1Over2));
        dst[i] = __float2bfloat16(y);
    }
}

__global__ void gelu_erf_f32_to_f16_kernel(const float* __restrict__ src,
                                           half* __restrict__ dst,
                                           int64_t total) {
    constexpr float kSqrt1Over2 = 0.70710678118654752440084436210485f;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        const float x = src[i];
        const float y = 0.5f * x * (1.0f + erff(x * kSqrt1Over2));
        dst[i] = __float2half(y);
    }
}

__global__ void add_bias_gelu_erf_bf16_inplace_kernel(nv_bfloat16* __restrict__ data,
                                                      const float* __restrict__ bias,
                                                      int rows,
                                                      int64_t total) {
    constexpr float kSqrt1Over2 = 0.70710678118654752440084436210485f;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        const int row = static_cast<int>(i % rows);
        const float x = __bfloat162float(data[i]) + (bias != nullptr ? bias[row] : 0.0f);
        const float y = 0.5f * x * (1.0f + erff(x * kSqrt1Over2));
        data[i] = __float2bfloat16(y);
    }
}

__global__ void bf16_wmma_matmul_bias_kernel(const nv_bfloat16* __restrict__ weights,
                                             const nv_bfloat16* __restrict__ input,
                                             const float* __restrict__ bias,
                                             float* __restrict__ output,
                                             int k,
                                             int rows,
                                             int ncols) {
    constexpr int tile = 16;
    constexpr int warps_per_cta = 1;
    const int warp_id = 0;
    const int row0 = blockIdx.y * tile;
    const int col0 = blockIdx.x * tile;
    const int batch = blockIdx.z;
    const int col_batch0 = batch * ncols + col0;

    wmma::fragment<wmma::matrix_a, tile, tile, tile, nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, tile, tile, tile, nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, tile, tile, tile, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < k; kk += tile) {
        const nv_bfloat16* a = weights + row0 * k + kk;
        const nv_bfloat16* b = input + col_batch0 * k + kk;
        wmma::load_matrix_sync(a_frag, a, k);
        wmma::load_matrix_sync(b_frag, b, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float tile_out[warps_per_cta * tile * tile];
    float* warp_tile_out = tile_out + warp_id * tile * tile;
    wmma::store_matrix_sync(warp_tile_out, c_frag, tile, wmma::mem_row_major);
    __syncwarp();

    for (int i = threadIdx.x; i < tile * tile; i += warpSize) {
        const int r = i / tile;
        const int c = i % tile;
        const int row = row0 + r;
        const int col = col0 + c;
        if (row < rows && col < ncols) {
            output[(batch * ncols + col) * rows + row] =
                warp_tile_out[i] + (bias != nullptr ? bias[row] : 0.0f);
        }
    }
}

__global__ void bf16_wmma_matmul_bias_gelu_bf16_kernel(const nv_bfloat16* __restrict__ weights,
                                                       const nv_bfloat16* __restrict__ input,
                                                       const float* __restrict__ bias,
                                                       nv_bfloat16* __restrict__ output,
                                                       int k,
                                                       int rows,
                                                       int ncols) {
    constexpr int tile = 16;
    constexpr int warps_per_cta = 1;
    const int warp_id = 0;
    const int row0 = blockIdx.y * tile;
    const int col0 = blockIdx.x * tile;
    const int batch = blockIdx.z;
    const int col_batch0 = batch * ncols + col0;

    wmma::fragment<wmma::matrix_a, tile, tile, tile, nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, tile, tile, tile, nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, tile, tile, tile, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < k; kk += tile) {
        const nv_bfloat16* a = weights + row0 * k + kk;
        const nv_bfloat16* b = input + col_batch0 * k + kk;
        wmma::load_matrix_sync(a_frag, a, k);
        wmma::load_matrix_sync(b_frag, b, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float tile_out[warps_per_cta * tile * tile];
    float* warp_tile_out = tile_out + warp_id * tile * tile;
    wmma::store_matrix_sync(warp_tile_out, c_frag, tile, wmma::mem_row_major);
    __syncwarp();

    for (int i = threadIdx.x; i < tile * tile; i += warpSize) {
        const int r = i / tile;
        const int c = i % tile;
        const int row = row0 + r;
        const int col = col0 + c;
        if (row < rows && col < ncols) {
            const float x = warp_tile_out[i] + (bias != nullptr ? bias[row] : 0.0f);
            output[(batch * ncols + col) * rows + row] = __float2bfloat16(gelu_tanh_f32(x));
        }
    }
}

__global__ void gelu_kernel(const float* __restrict__ input,
                            const float* __restrict__ bias,
                            float* __restrict__ output,
                            int rows,
                            int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        const float bias_v = bias != nullptr ? bias[i % rows] : 0.0f;
        output[i] = gelu_tanh_f32(input[i] + bias_v);
    }
}

__global__ void quantize_ds4_kernel(const float* __restrict__ input,
                                    void* __restrict__ output,
                                    int rows,
                                    int padded_rows,
                                    int cols) {
    const int row0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (row0 >= padded_rows) {
        return;
    }

    const int col = blockIdx.x;
    const int iqs = row0 % (4 * kQk8_1);
    const int base = col * rows + row0;
    const float4 xi = make_float4(row0 + 0 < rows ? input[base + 0] : 0.0f,
                                  row0 + 1 < rows ? input[base + 1] : 0.0f,
                                  row0 + 2 < rows ? input[base + 2] : 0.0f,
                                  row0 + 3 < rows ? input[base + 3] : 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
    }

    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const char4 q = make_char4(static_cast<int8_t>(roundf(xi.x * d_inv)),
                               static_cast<int8_t>(roundf(xi.y * d_inv)),
                               static_cast<int8_t>(roundf(xi.z * d_inv)),
                               static_cast<int8_t>(roundf(xi.w * d_inv)));

    auto* y = static_cast<BlockQ8_1MmqDs4*>(output);
    const int block = (row0 / (4 * kQk8_1)) * cols + col;
    reinterpret_cast<char4*>(y[block].qs)[iqs / 4] = q;

    if (iqs % 32 == 0) {
        const float d = d_inv != 0.0f ? 1.0f / d_inv : 0.0f;
        y[block].ds4[iqs / 32] = make_half2(d, sum);
    }
}

__global__ void gelu_quantize_ds4_kernel(const float* __restrict__ input,
                                         const float* __restrict__ bias,
                                         float* __restrict__ gelu_output,
                                         void* __restrict__ quant_output,
                                         int rows,
                                         int padded_rows,
                                         int cols) {
    const int row0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (row0 >= padded_rows) {
        return;
    }

    const int col = blockIdx.x;
    const int base = col * rows + row0;
    const int iqs = row0 % (4 * kQk8_1);
    const float4 bias_v = bias != nullptr ? make_float4(row0 + 0 < rows ? bias[row0 + 0] : 0.0f,
                                                        row0 + 1 < rows ? bias[row0 + 1] : 0.0f,
                                                        row0 + 2 < rows ? bias[row0 + 2] : 0.0f,
                                                        row0 + 3 < rows ? bias[row0 + 3] : 0.0f)
                                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 x = make_float4(row0 + 0 < rows ? input[base + 0] : 0.0f,
                                 row0 + 1 < rows ? input[base + 1] : 0.0f,
                                 row0 + 2 < rows ? input[base + 2] : 0.0f,
                                 row0 + 3 < rows ? input[base + 3] : 0.0f);
    const float4 xi = make_float4(gelu_tanh_f32(x.x + bias_v.x),
                                  gelu_tanh_f32(x.y + bias_v.y),
                                  gelu_tanh_f32(x.z + bias_v.z),
                                  gelu_tanh_f32(x.w + bias_v.w));
    if (row0 + 0 < rows) {
        gelu_output[base + 0] = xi.x;
    }
    if (row0 + 1 < rows) {
        gelu_output[base + 1] = xi.y;
    }
    if (row0 + 2 < rows) {
        gelu_output[base + 2] = xi.z;
    }
    if (row0 + 3 < rows) {
        gelu_output[base + 3] = xi.w;
    }

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
    }

    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const char4 q = make_char4(static_cast<int8_t>(roundf(xi.x * d_inv)),
                               static_cast<int8_t>(roundf(xi.y * d_inv)),
                               static_cast<int8_t>(roundf(xi.z * d_inv)),
                               static_cast<int8_t>(roundf(xi.w * d_inv)));

    auto* y = static_cast<BlockQ8_1MmqDs4*>(quant_output);
    const int block = (row0 / (4 * kQk8_1)) * cols + col;
    reinterpret_cast<char4*>(y[block].qs)[iqs / 4] = q;

    if (iqs % 32 == 0) {
        const float d = d_inv != 0.0f ? 1.0f / d_inv : 0.0f;
        y[block].ds4[iqs / 32] = make_half2(d, sum);
    }
}

struct DeviceBuffer {
    void* ptr = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

[[nodiscard]] bool alloc(DeviceBuffer& buffer, size_t bytes) {
    if (bytes == 0) {
        buffer.ptr = nullptr;
        return true;
    }
    return cudaMalloc(&buffer.ptr, bytes) == cudaSuccess;
}

[[nodiscard]] bool ok(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS;
}

[[nodiscard]] double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms);
}

[[nodiscard]] Sam3F32BatchedCustomResult summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const auto at_quantile = [&](double q) {
        const size_t idx =
            std::min(samples.size() - 1, static_cast<size_t>(std::ceil(q * samples.size())) - 1);
        return samples[idx];
    };

    return Sam3F32BatchedCustomResult{
        .mean_ms = sum / static_cast<double>(samples.size()),
        .median_ms = at_quantile(0.50),
        .p95_ms = at_quantile(0.95),
        .min_ms = samples.front(),
        .max_ms = samples.back(),
    };
}

}  // namespace

int sam3_f32_batched_custom_bench(const float* weights,
                                  const float* input,
                                  const float* bias,
                                  float* output,
                                  int64_t k,
                                  int64_t rows,
                                  int64_t cols,
                                  int64_t batches,
                                  int warmup,
                                  int iters,
                                  Sam3F32BatchedCustomResult* result) {
    if (weights == nullptr || input == nullptr || output == nullptr || result == nullptr ||
        k <= 0 || rows <= 0 || cols <= 0 || batches <= 0 || warmup < 0 || iters <= 0 ||
        k > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols * batches > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return 2;
    }

    const size_t weight_bytes = static_cast<size_t>(k * rows) * sizeof(float);
    const size_t input_bytes = static_cast<size_t>(k * cols * batches) * sizeof(float);
    const size_t bias_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows * cols * batches) * sizeof(float);

    DeviceBuffer d_weights;
    DeviceBuffer d_input;
    DeviceBuffer d_bias;
    DeviceBuffer d_output;
    if (!alloc(d_weights, weight_bytes) || !alloc(d_input, input_bytes) ||
        !alloc(d_output, output_bytes) || (bias != nullptr && !alloc(d_bias, bias_bytes))) {
        return 1;
    }

    if (cudaMemcpy(d_weights.ptr, weights, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_input.ptr, input, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        (bias != nullptr &&
         cudaMemcpy(d_bias.ptr, bias, bias_bytes, cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        return 1;
    }

    const int ncols = static_cast<int>(cols * batches);
    const dim3 block(kTile, kTile);
    const dim3 grid((ncols + kTile - 1) / kTile, (static_cast<int>(rows) + kTile - 1) / kTile);
    auto launch = [&] {
        f32_batched_matmul_bias_kernel<<<grid, block>>>(static_cast<const float*>(d_weights.ptr),
                                                        static_cast<const float*>(d_input.ptr),
                                                        static_cast<const float*>(d_bias.ptr),
                                                        static_cast<float*>(d_output.ptr),
                                                        static_cast<int>(k),
                                                        static_cast<int>(rows),
                                                        ncols);
    };

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    const bool ok =
        cudaGetLastError() == cudaSuccess &&
        cudaMemcpy(output, d_output.ptr, output_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!ok) {
        return 1;
    }

    *result = summarize(std::move(samples));
    return 0;
}

int sam3_gelu_quant_ds4_bench(const float* input,
                              const float* bias,
                              int64_t rows,
                              int64_t cols,
                              int warmup,
                              int iters,
                              Sam3GeluQuantDs4BenchResult* result) {
    if (input == nullptr || result == nullptr || rows <= 0 || cols <= 0 || warmup < 0 ||
        iters <= 0 || rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) || rows % 4 != 0) {
        return 2;
    }

    const int64_t padded_rows = ((rows + 4 * kQk8_1 - 1) / (4 * kQk8_1)) * (4 * kQk8_1);
    const size_t input_bytes = static_cast<size_t>(rows * cols) * sizeof(float);
    const size_t quant_bytes =
        static_cast<size_t>((padded_rows / (4 * kQk8_1)) * cols) * sizeof(BlockQ8_1MmqDs4);

    DeviceBuffer d_input;
    DeviceBuffer d_bias;
    DeviceBuffer d_separate_gelu;
    DeviceBuffer d_fused_gelu;
    DeviceBuffer d_separate_quant;
    DeviceBuffer d_fused_quant;
    if (!alloc(d_input, input_bytes) ||
        (bias != nullptr && !alloc(d_bias, static_cast<size_t>(rows) * sizeof(float))) ||
        !alloc(d_separate_gelu, input_bytes) || !alloc(d_fused_gelu, input_bytes) ||
        !alloc(d_separate_quant, quant_bytes) || !alloc(d_fused_quant, quant_bytes)) {
        return 1;
    }

    if (cudaMemcpy(d_input.ptr, input, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        (bias != nullptr && cudaMemcpy(d_bias.ptr,
                                       bias,
                                       static_cast<size_t>(rows) * sizeof(float),
                                       cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        return 1;
    }

    const int rows_i = static_cast<int>(rows);
    const int padded_rows_i = static_cast<int>(padded_rows);
    const int cols_i = static_cast<int>(cols);
    const int64_t total = rows * cols;
    const dim3 gelu_block(256);
    const dim3 gelu_grid(static_cast<unsigned>((total + gelu_block.x - 1) / gelu_block.x));
    const dim3 quant_block(kQuantBlockSizeMmq);
    const dim3 quant_grid(
        cols_i, static_cast<unsigned>((padded_rows + 4 * quant_block.x - 1) / (4 * quant_block.x)));

    auto launch_separate = [&] {
        gelu_kernel<<<gelu_grid, gelu_block>>>(static_cast<const float*>(d_input.ptr),
                                               static_cast<const float*>(d_bias.ptr),
                                               static_cast<float*>(d_separate_gelu.ptr),
                                               rows_i,
                                               total);
        quantize_ds4_kernel<<<quant_grid, quant_block>>>(
            static_cast<const float*>(d_separate_gelu.ptr),
            d_separate_quant.ptr,
            rows_i,
            padded_rows_i,
            cols_i);
    };
    auto launch_fused = [&] {
        gelu_quantize_ds4_kernel<<<quant_grid, quant_block>>>(
            static_cast<const float*>(d_input.ptr),
            static_cast<const float*>(d_bias.ptr),
            static_cast<float*>(d_fused_gelu.ptr),
            d_fused_quant.ptr,
            rows_i,
            padded_rows_i,
            cols_i);
    };

    for (int i = 0; i < warmup; ++i) {
        launch_separate();
        launch_fused();
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }

    std::vector<double> separate_samples;
    std::vector<double> fused_samples;
    separate_samples.reserve(static_cast<size_t>(iters));
    fused_samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch_separate();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        separate_samples.push_back(elapsed_ms(start, stop));

        cudaEventRecord(start);
        launch_fused();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        fused_samples.push_back(elapsed_ms(start, stop));
    }

    std::vector<float> separate_gelu(static_cast<size_t>(rows * cols));
    std::vector<float> fused_gelu(static_cast<size_t>(rows * cols));
    std::vector<uint8_t> separate_quant(quant_bytes);
    std::vector<uint8_t> fused_quant(quant_bytes);
    const bool copy_ok =
        cudaGetLastError() == cudaSuccess &&
        cudaMemcpy(
            separate_gelu.data(), d_separate_gelu.ptr, input_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess &&
        cudaMemcpy(fused_gelu.data(), d_fused_gelu.ptr, input_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess &&
        cudaMemcpy(
            separate_quant.data(), d_separate_quant.ptr, quant_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess &&
        cudaMemcpy(fused_quant.data(), d_fused_quant.ptr, quant_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!copy_ok) {
        return 1;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < separate_gelu.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(separate_gelu[i] - fused_gelu[i]));
    }

    int diff_bytes = 0;
    for (size_t i = 0; i < separate_quant.size(); ++i) {
        diff_bytes += separate_quant[i] != fused_quant[i] ? 1 : 0;
    }

    *result = Sam3GeluQuantDs4BenchResult{
        .separate = summarize(std::move(separate_samples)),
        .fused = summarize(std::move(fused_samples)),
        .max_abs_output_delta = max_abs,
        .q8_diff_bytes = diff_bytes,
    };
    return 0;
}

int sam3_bf16_gelu_cast_bench(const float* input,
                              float* output,
                              int64_t rows,
                              int64_t cols,
                              int64_t batches,
                              int warmup,
                              int iters,
                              Sam3Bf16GeluCastResult* result) {
    if (input == nullptr || output == nullptr || result == nullptr || rows <= 0 || cols <= 0 ||
        batches <= 0 || warmup < 0 || iters <= 0 ||
        rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return 2;
    }

    const int64_t total = rows * cols * batches;
    const size_t input_bytes = static_cast<size_t>(total) * sizeof(float);
    const size_t output_bf16_bytes = static_cast<size_t>(total) * sizeof(nv_bfloat16);
    const size_t output_f32_bytes = static_cast<size_t>(total) * sizeof(float);

    DeviceBuffer d_input;
    DeviceBuffer d_output_bf16;
    DeviceBuffer d_output_f32;
    if (!alloc(d_input, input_bytes) || !alloc(d_output_bf16, output_bf16_bytes) ||
        !alloc(d_output_f32, output_f32_bytes)) {
        return 1;
    }
    if (cudaMemcpy(d_input.ptr, input, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        return 1;
    }

    constexpr int block = 256;
    const auto grid = static_cast<unsigned>((total + block - 1) / block);
    auto launch = [&] {
        gelu_erf_f32_to_bf16_kernel<<<grid, block>>>(static_cast<const float*>(d_input.ptr),
                                                     static_cast<nv_bfloat16*>(d_output_bf16.ptr),
                                                     total);
    };

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    bf16_to_f32_kernel<<<grid, block>>>(static_cast<const nv_bfloat16*>(d_output_bf16.ptr),
                                        static_cast<float*>(d_output_f32.ptr),
                                        total);
    const bool copy_ok =
        cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, d_output_f32.ptr, output_f32_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!copy_ok) {
        return 1;
    }

    double checksum = 0.0;
    const int64_t stride = std::max<int64_t>(1, total / 4096);
    for (int64_t i = 0; i < total; i += stride) {
        checksum += static_cast<double>(output[i]);
    }

    *result = Sam3Bf16GeluCastResult{
        .gelu_cast = summarize(std::move(samples)),
        .output_checksum = checksum,
    };
    return 0;
}

int sam3_f32_batched_cublaslt_bench(const float* weights,
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
                                    Sam3F32BatchedCustomResult* result) {
    if (weights == nullptr || input == nullptr || output == nullptr || result == nullptr ||
        k <= 0 || rows <= 0 || cols <= 0 || batches <= 0 || warmup < 0 || iters <= 0 ||
        k > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return 2;
    }

    const size_t weight_bytes = static_cast<size_t>(k * rows) * sizeof(float);
    const size_t input_bytes = static_cast<size_t>(k * cols * batches) * sizeof(float);
    const size_t bias_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows * cols * batches) * sizeof(float);

    DeviceBuffer d_weights;
    DeviceBuffer d_input;
    DeviceBuffer d_bias;
    DeviceBuffer d_output;
    DeviceBuffer workspace;
    constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
    if (!alloc(d_weights, weight_bytes) || !alloc(d_input, input_bytes) ||
        !alloc(d_output, output_bytes) || !alloc(workspace, kWorkspaceBytes) ||
        (bias != nullptr && !alloc(d_bias, bias_bytes))) {
        return 1;
    }

    if (cudaMemcpy(d_weights.ptr, weights, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_input.ptr, input, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        (bias != nullptr &&
         cudaMemcpy(d_bias.ptr, bias, bias_bytes, cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&] {
        if (pref != nullptr) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (op_desc != nullptr) {
            cublasLtMatmulDescDestroy(op_desc);
        }
        if (lt != nullptr) {
            cublasLtDestroy(lt);
        }
    };

    const cublasComputeType_t compute_type =
        fast_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    if (!ok(cublasLtCreate(&lt)) ||
        !ok(cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatrixLayoutCreate(&a_desc,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(k),
                                       static_cast<uint64_t>(rows),
                                       static_cast<int64_t>(k))) ||
        !ok(cublasLtMatrixLayoutCreate(&b_desc,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(k),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(k))) ||
        !ok(cublasLtMatrixLayoutCreate(&c_desc,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(rows),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(rows))) ||
        !ok(cublasLtMatmulPreferenceCreate(&pref))) {
        cleanup();
        return 1;
    }

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    if (!ok(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)))) {
        cleanup();
        return 1;
    }
    if (bias != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const void* bias_ptr = d_bias.ptr;
        if (!ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
            !ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)))) {
            cleanup();
            return 1;
        }
    }

    const int32_t batch_count = static_cast<int32_t>(batches);
    const int64_t a_stride = 0;
    const int64_t b_stride = k * cols;
    const int64_t c_stride = rows * cols;
    if (!ok(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)))) {
        cleanup();
        return 1;
    }

    const size_t workspace_bytes = kWorkspaceBytes;
    if (!ok(cublasLtMatmulPreferenceSetAttribute(pref,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_bytes,
                                                 sizeof(workspace_bytes)))) {
        cleanup();
        return 1;
    }

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(
            lt, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned)) ||
        returned == 0) {
        cleanup();
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        cleanup();
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch = [&] {
        return cublasLtMatmul(lt,
                              op_desc,
                              &alpha,
                              d_weights.ptr,
                              a_desc,
                              d_input.ptr,
                              b_desc,
                              &beta,
                              d_output.ptr,
                              c_desc,
                              d_output.ptr,
                              c_desc,
                              &heuristic.algo,
                              workspace.ptr,
                              workspace_bytes,
                              nullptr);
    };

    for (int i = 0; i < warmup; ++i) {
        if (!ok(launch())) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cleanup();
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        if (!ok(launch())) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    const bool copy_ok =
        cudaMemcpy(output, d_output.ptr, output_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    if (!copy_ok) {
        return 1;
    }

    *result = summarize(std::move(samples));
    return 0;
}

int sam3_bf16_batched_cublaslt_bench(const float* weights,
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
                                     Sam3F32BatchedCustomResult* result) {
    if (weights == nullptr || input == nullptr || output == nullptr || result == nullptr ||
        k <= 0 || rows <= 0 || cols <= 0 || batches <= 0 || warmup < 0 || iters <= 0 ||
        k > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return 2;
    }

    const int64_t input_elems = k * cols * batches;
    const int64_t weight_elems = k * rows;
    const int64_t output_elems = rows * cols * batches;
    const size_t weight_f32_bytes = static_cast<size_t>(weight_elems) * sizeof(float);
    const size_t input_f32_bytes = static_cast<size_t>(input_elems) * sizeof(float);
    const size_t weight_bf16_bytes = static_cast<size_t>(weight_elems) * sizeof(nv_bfloat16);
    const size_t input_bf16_bytes = static_cast<size_t>(input_elems) * sizeof(nv_bfloat16);
    const size_t bias_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t output_f32_bytes = static_cast<size_t>(output_elems) * sizeof(float);
    const size_t output_bf16_bytes = static_cast<size_t>(output_elems) * sizeof(nv_bfloat16);

    DeviceBuffer d_weights_f32;
    DeviceBuffer d_input_f32;
    DeviceBuffer d_weights_bf16;
    DeviceBuffer d_input_bf16;
    DeviceBuffer d_bias;
    DeviceBuffer d_output_f32;
    DeviceBuffer d_output_bf16;
    DeviceBuffer workspace;
    const size_t workspace_bytes = options.workspace_bytes;
    if (!alloc(d_weights_f32, weight_f32_bytes) || !alloc(d_input_f32, input_f32_bytes) ||
        !alloc(d_weights_bf16, weight_bf16_bytes) || !alloc(d_input_bf16, input_bf16_bytes) ||
        !alloc(workspace, workspace_bytes) ||
        (options.direct_bf16_dst ? !alloc(d_output_bf16, output_bf16_bytes)
                                 : !alloc(d_output_f32, output_f32_bytes)) ||
        (options.direct_bf16_dst && options.c_f32_for_direct_dst &&
         !alloc(d_output_f32, output_f32_bytes)) ||
        (bias != nullptr && !alloc(d_bias, bias_bytes))) {
        return 1;
    }

    if (cudaMemcpy(d_weights_f32.ptr, weights, weight_f32_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_input_f32.ptr, input, input_f32_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        (bias != nullptr &&
         cudaMemcpy(d_bias.ptr, bias, bias_bytes, cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    constexpr int convert_block = 256;
    f32_to_bf16_kernel<<<static_cast<unsigned>((weight_elems + convert_block - 1) / convert_block),
                         convert_block>>>(static_cast<const float*>(d_weights_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_weights_bf16.ptr),
                                          weight_elems);
    f32_to_bf16_kernel<<<static_cast<unsigned>((input_elems + convert_block - 1) / convert_block),
                         convert_block>>>(static_cast<const float*>(d_input_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_input_bf16.ptr),
                                          input_elems);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&] {
        if (pref != nullptr) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        if (d_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(d_desc);
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (op_desc != nullptr) {
            cublasLtMatmulDescDestroy(op_desc);
        }
        if (lt != nullptr) {
            cublasLtDestroy(lt);
        }
    };

    const cublasComputeType_t compute_type = options.compute == Sam3Bf16CublasLtCompute::fast_16bf
                                                 ? CUBLAS_COMPUTE_32F_FAST_16BF
                                                 : CUBLAS_COMPUTE_32F;
    const cudaDataType_t dst_type = options.direct_bf16_dst ? CUDA_R_16BF : CUDA_R_32F;
    const cudaDataType_t c_type =
        options.direct_bf16_dst && options.c_f32_for_direct_dst ? CUDA_R_32F : dst_type;
    if (!ok(cublasLtCreate(&lt)) ||
        !ok(cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatrixLayoutCreate(&a_desc,
                                       CUDA_R_16BF,
                                       static_cast<uint64_t>(k),
                                       static_cast<uint64_t>(rows),
                                       static_cast<int64_t>(k))) ||
        !ok(cublasLtMatrixLayoutCreate(&b_desc,
                                       CUDA_R_16BF,
                                       static_cast<uint64_t>(k),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(k))) ||
        !ok(cublasLtMatrixLayoutCreate(&c_desc,
                                       c_type,
                                       static_cast<uint64_t>(rows),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(rows))) ||
        !ok(cublasLtMatrixLayoutCreate(&d_desc,
                                       dst_type,
                                       static_cast<uint64_t>(rows),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(rows))) ||
        !ok(cublasLtMatmulPreferenceCreate(&pref))) {
        cleanup();
        return 1;
    }

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    if (!ok(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)))) {
        cleanup();
        return 1;
    }
    if (bias != nullptr) {
        const cublasLtEpilogue_t epilogue =
            options.gelu_epilogue ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
        const void* bias_ptr = d_bias.ptr;
        const cudaDataType_t bias_type = CUDA_R_32F;
        if (!ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
            !ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) ||
            !ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)))) {
            cleanup();
            return 1;
        }
    }

    const int32_t batch_count = static_cast<int32_t>(batches);
    const int64_t a_stride = 0;
    const int64_t b_stride = k * cols;
    const int64_t c_stride = rows * cols;
    if (!ok(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            d_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            d_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)))) {
        cleanup();
        return 1;
    }

    if (!ok(cublasLtMatmulPreferenceSetAttribute(pref,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_bytes,
                                                 sizeof(workspace_bytes)))) {
        cleanup();
        return 1;
    }

    constexpr int max_heuristics = 16;
    cublasLtMatmulHeuristicResult_t heuristics[max_heuristics] = {};
    int returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(lt,
                                           op_desc,
                                           a_desc,
                                           b_desc,
                                           c_desc,
                                           d_desc,
                                           pref,
                                           max_heuristics,
                                           heuristics,
                                           &returned)) ||
        returned == 0) {
        cleanup();
        return 1;
    }
    if (options.algo_index < 0 || options.algo_index >= returned) {
        cleanup();
        return 3;
    }
    const auto algo = heuristics[options.algo_index].algo;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        cleanup();
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* d_output = options.direct_bf16_dst ? d_output_bf16.ptr : d_output_f32.ptr;
    void* c_input =
        options.direct_bf16_dst && options.c_f32_for_direct_dst ? d_output_f32.ptr : d_output;
    auto launch = [&] {
        return cublasLtMatmul(lt,
                              op_desc,
                              &alpha,
                              d_weights_bf16.ptr,
                              a_desc,
                              d_input_bf16.ptr,
                              b_desc,
                              &beta,
                              c_input,
                              c_desc,
                              d_output,
                              d_desc,
                              &algo,
                              workspace.ptr,
                              workspace_bytes,
                              nullptr);
    };

    for (int i = 0; i < warmup; ++i) {
        if (!ok(launch())) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cleanup();
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        if (!ok(launch())) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    bool copy_ok = false;
    if (options.direct_bf16_dst) {
        if (d_output_f32.ptr == nullptr && !alloc(d_output_f32, output_f32_bytes)) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
        bf16_to_f32_kernel<<<static_cast<unsigned>((output_elems + convert_block - 1) /
                                                   convert_block),
                             convert_block>>>(static_cast<const nv_bfloat16*>(d_output_bf16.ptr),
                                              static_cast<float*>(d_output_f32.ptr),
                                              output_elems);
        copy_ok = cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output, d_output_f32.ptr, output_f32_bytes, cudaMemcpyDeviceToHost) ==
                      cudaSuccess;
    } else {
        copy_ok = cudaMemcpy(output, d_output_f32.ptr, output_f32_bytes, cudaMemcpyDeviceToHost) ==
                  cudaSuccess;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    if (!copy_ok) {
        return 1;
    }

    *result = summarize(std::move(samples));
    return 0;
}

int sam3_f16_batched_cublaslt_bench(const float* weights,
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
                                    Sam3F32BatchedCustomResult* result) {
    if (weights == nullptr || input == nullptr || output == nullptr || result == nullptr ||
        k <= 0 || rows <= 0 || cols <= 0 || batches <= 0 || warmup < 0 || iters <= 0 ||
        k > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return 2;
    }

    const int64_t input_elems = k * cols * batches;
    const int64_t weight_elems = k * rows;
    const int64_t output_elems = rows * cols * batches;
    const size_t weight_f32_bytes = static_cast<size_t>(weight_elems) * sizeof(float);
    const size_t input_f32_bytes = static_cast<size_t>(input_elems) * sizeof(float);
    const size_t weight_f16_bytes = static_cast<size_t>(weight_elems) * sizeof(half);
    const size_t input_f16_bytes = static_cast<size_t>(input_elems) * sizeof(half);
    const size_t bias_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t output_f32_bytes = static_cast<size_t>(output_elems) * sizeof(float);

    DeviceBuffer d_weights_f32;
    DeviceBuffer d_input_f32;
    DeviceBuffer d_weights_f16;
    DeviceBuffer d_input_f16;
    DeviceBuffer d_bias;
    DeviceBuffer d_output_f32;
    DeviceBuffer workspace;
    const size_t workspace_bytes = options.workspace_bytes;
    if (!alloc(d_weights_f32, weight_f32_bytes) || !alloc(d_input_f32, input_f32_bytes) ||
        !alloc(d_weights_f16, weight_f16_bytes) || !alloc(d_input_f16, input_f16_bytes) ||
        !alloc(d_output_f32, output_f32_bytes) || !alloc(workspace, workspace_bytes) ||
        (bias != nullptr && !alloc(d_bias, bias_bytes))) {
        return 1;
    }

    if (cudaMemcpy(d_weights_f32.ptr, weights, weight_f32_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_input_f32.ptr, input, input_f32_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        (bias != nullptr &&
         cudaMemcpy(d_bias.ptr, bias, bias_bytes, cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    constexpr int convert_block = 256;
    f32_to_f16_kernel<<<static_cast<unsigned>((weight_elems + convert_block - 1) / convert_block),
                        convert_block>>>(static_cast<const float*>(d_weights_f32.ptr),
                                         static_cast<half*>(d_weights_f16.ptr),
                                         weight_elems);
    f32_to_f16_kernel<<<static_cast<unsigned>((input_elems + convert_block - 1) / convert_block),
                        convert_block>>>(static_cast<const float*>(d_input_f32.ptr),
                                         static_cast<half*>(d_input_f16.ptr),
                                         input_elems);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&] {
        if (pref != nullptr) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (op_desc != nullptr) {
            cublasLtMatmulDescDestroy(op_desc);
        }
        if (lt != nullptr) {
            cublasLtDestroy(lt);
        }
    };

    const cublasComputeType_t compute_type =
        options.compute == Sam3F16CublasLtCompute::compute_16f
            ? CUBLAS_COMPUTE_16F
            : (options.compute == Sam3F16CublasLtCompute::fast_16f ? CUBLAS_COMPUTE_32F_FAST_16F
                                                                   : CUBLAS_COMPUTE_32F);
    if (!ok(cublasLtCreate(&lt)) ||
        !ok(cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatrixLayoutCreate(&a_desc,
                                       CUDA_R_16F,
                                       static_cast<uint64_t>(k),
                                       static_cast<uint64_t>(rows),
                                       static_cast<int64_t>(k))) ||
        !ok(cublasLtMatrixLayoutCreate(&b_desc,
                                       CUDA_R_16F,
                                       static_cast<uint64_t>(k),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(k))) ||
        !ok(cublasLtMatrixLayoutCreate(&c_desc,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(rows),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(rows))) ||
        !ok(cublasLtMatmulPreferenceCreate(&pref))) {
        cleanup();
        return 1;
    }

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    if (!ok(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)))) {
        cleanup();
        return 1;
    }
    if (bias != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const void* bias_ptr = d_bias.ptr;
        const cudaDataType_t bias_type = CUDA_R_32F;
        if (!ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
            !ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) ||
            !ok(cublasLtMatmulDescSetAttribute(
                op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)))) {
            cleanup();
            return 1;
        }
    }

    const int32_t batch_count = static_cast<int32_t>(batches);
    const int64_t a_stride = 0;
    const int64_t b_stride = k * cols;
    const int64_t c_stride = rows * cols;
    if (!ok(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)))) {
        cleanup();
        return 1;
    }

    if (!ok(cublasLtMatmulPreferenceSetAttribute(pref,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_bytes,
                                                 sizeof(workspace_bytes)))) {
        cleanup();
        return 1;
    }

    constexpr int max_heuristics = 16;
    cublasLtMatmulHeuristicResult_t heuristics[max_heuristics] = {};
    int returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(lt,
                                           op_desc,
                                           a_desc,
                                           b_desc,
                                           c_desc,
                                           c_desc,
                                           pref,
                                           max_heuristics,
                                           heuristics,
                                           &returned)) ||
        returned == 0) {
        cleanup();
        return 1;
    }
    if (options.algo_index < 0 || options.algo_index >= returned) {
        cleanup();
        return 3;
    }
    const auto algo = heuristics[options.algo_index].algo;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        cleanup();
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch = [&] {
        return cublasLtMatmul(lt,
                              op_desc,
                              &alpha,
                              d_weights_f16.ptr,
                              a_desc,
                              d_input_f16.ptr,
                              b_desc,
                              &beta,
                              d_output_f32.ptr,
                              c_desc,
                              d_output_f32.ptr,
                              c_desc,
                              &algo,
                              workspace.ptr,
                              workspace_bytes,
                              nullptr);
    };

    for (int i = 0; i < warmup; ++i) {
        if (!ok(launch())) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cleanup();
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        if (!ok(launch())) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    const bool copy_ok =
        cudaMemcpy(output, d_output_f32.ptr, output_f32_bytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    if (!copy_ok) {
        return 1;
    }

    *result = summarize(std::move(samples));
    return 0;
}

int sam3_f16_mlp_chain_cublaslt_bench(const float* fc1_weights,
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
                                      Sam3F16MlpChainResult* result) {
    if (fc1_weights == nullptr || fc2_weights == nullptr || input == nullptr || output == nullptr ||
        result == nullptr || input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0 || cols <= 0 ||
        batches <= 0 || warmup < 0 || iters <= 0 ||
        input_dim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        hidden_dim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        output_dim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return 2;
    }

    const int64_t ncols_total = cols * batches;
    const int64_t input_elems = input_dim * ncols_total;
    const int64_t fc1_weight_elems = input_dim * hidden_dim;
    const int64_t fc2_weight_elems = hidden_dim * output_dim;
    const int64_t fc1_elems = hidden_dim * ncols_total;
    const int64_t output_elems = output_dim * ncols_total;

    DeviceBuffer d_fc1_weights_f32;
    DeviceBuffer d_fc2_weights_f32;
    DeviceBuffer d_input_f32;
    DeviceBuffer d_fc1_weights_f16;
    DeviceBuffer d_fc2_weights_f16;
    DeviceBuffer d_input_f16;
    DeviceBuffer d_fc1_bias;
    DeviceBuffer d_fc2_bias;
    DeviceBuffer d_fc1_f32;
    DeviceBuffer d_fc1_f16;
    DeviceBuffer d_output_f32;
    DeviceBuffer workspace;

    if (!alloc(d_fc1_weights_f32, static_cast<size_t>(fc1_weight_elems) * sizeof(float)) ||
        !alloc(d_fc2_weights_f32, static_cast<size_t>(fc2_weight_elems) * sizeof(float)) ||
        !alloc(d_input_f32, static_cast<size_t>(input_elems) * sizeof(float)) ||
        !alloc(d_fc1_weights_f16, static_cast<size_t>(fc1_weight_elems) * sizeof(half)) ||
        !alloc(d_fc2_weights_f16, static_cast<size_t>(fc2_weight_elems) * sizeof(half)) ||
        !alloc(d_input_f16, static_cast<size_t>(input_elems) * sizeof(half)) ||
        (fc1_bias != nullptr &&
         !alloc(d_fc1_bias, static_cast<size_t>(hidden_dim) * sizeof(float))) ||
        (fc2_bias != nullptr &&
         !alloc(d_fc2_bias, static_cast<size_t>(output_dim) * sizeof(float))) ||
        !alloc(d_fc1_f32, static_cast<size_t>(fc1_elems) * sizeof(float)) ||
        !alloc(d_fc1_f16, static_cast<size_t>(fc1_elems) * sizeof(half)) ||
        !alloc(d_output_f32, static_cast<size_t>(output_elems) * sizeof(float)) ||
        !alloc(workspace, options.workspace_bytes)) {
        return 1;
    }

    if (cudaMemcpy(d_fc1_weights_f32.ptr,
                   fc1_weights,
                   static_cast<size_t>(fc1_weight_elems) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_fc2_weights_f32.ptr,
                   fc2_weights,
                   static_cast<size_t>(fc2_weight_elems) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_input_f32.ptr,
                   input,
                   static_cast<size_t>(input_elems) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (fc1_bias != nullptr && cudaMemcpy(d_fc1_bias.ptr,
                                           fc1_bias,
                                           static_cast<size_t>(hidden_dim) * sizeof(float),
                                           cudaMemcpyHostToDevice) != cudaSuccess) ||
        (fc2_bias != nullptr && cudaMemcpy(d_fc2_bias.ptr,
                                           fc2_bias,
                                           static_cast<size_t>(output_dim) * sizeof(float),
                                           cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    constexpr int convert_block = 256;
    f32_to_f16_kernel<<<static_cast<unsigned>((fc1_weight_elems + convert_block - 1) /
                                              convert_block),
                        convert_block>>>(static_cast<const float*>(d_fc1_weights_f32.ptr),
                                         static_cast<half*>(d_fc1_weights_f16.ptr),
                                         fc1_weight_elems);
    f32_to_f16_kernel<<<static_cast<unsigned>((fc2_weight_elems + convert_block - 1) /
                                              convert_block),
                        convert_block>>>(static_cast<const float*>(d_fc2_weights_f32.ptr),
                                         static_cast<half*>(d_fc2_weights_f16.ptr),
                                         fc2_weight_elems);
    f32_to_f16_kernel<<<static_cast<unsigned>((input_elems + convert_block - 1) / convert_block),
                        convert_block>>>(static_cast<const float*>(d_input_f32.ptr),
                                         static_cast<half*>(d_input_f16.ptr),
                                         input_elems);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t fc1_op = nullptr;
    cublasLtMatmulDesc_t fc2_op = nullptr;
    cublasLtMatrixLayout_t fc1_a = nullptr;
    cublasLtMatrixLayout_t fc1_b = nullptr;
    cublasLtMatrixLayout_t fc1_c = nullptr;
    cublasLtMatrixLayout_t fc2_a = nullptr;
    cublasLtMatrixLayout_t fc2_b = nullptr;
    cublasLtMatrixLayout_t fc2_c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&] {
        if (pref != nullptr) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        for (auto* layout : {fc2_c, fc2_b, fc2_a, fc1_c, fc1_b, fc1_a}) {
            if (layout != nullptr) {
                cublasLtMatrixLayoutDestroy(layout);
            }
        }
        if (fc2_op != nullptr) {
            cublasLtMatmulDescDestroy(fc2_op);
        }
        if (fc1_op != nullptr) {
            cublasLtMatmulDescDestroy(fc1_op);
        }
        if (lt != nullptr) {
            cublasLtDestroy(lt);
        }
    };

    const cublasComputeType_t compute_type =
        options.compute == Sam3F16CublasLtCompute::compute_16f
            ? CUBLAS_COMPUTE_16F
            : (options.compute == Sam3F16CublasLtCompute::fast_16f ? CUBLAS_COMPUTE_32F_FAST_16F
                                                                   : CUBLAS_COMPUTE_32F);
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    if (!ok(cublasLtCreate(&lt)) ||
        !ok(cublasLtMatmulDescCreate(&fc1_op, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatmulDescCreate(&fc2_op, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc1_op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc1_op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc2_op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc2_op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)))) {
        cleanup();
        return 1;
    }

    if (fc1_bias != nullptr || fc2_bias != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const cudaDataType_t bias_type = CUDA_R_32F;
        if (fc1_bias != nullptr) {
            const void* fc1_bias_ptr = d_fc1_bias.ptr;
            if (!ok(cublasLtMatmulDescSetAttribute(
                    fc1_op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
                !ok(cublasLtMatmulDescSetAttribute(
                    fc1_op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) ||
                !ok(cublasLtMatmulDescSetAttribute(fc1_op,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &fc1_bias_ptr,
                                                   sizeof(fc1_bias_ptr)))) {
                cleanup();
                return 1;
            }
        }
        if (fc2_bias != nullptr) {
            const void* fc2_bias_ptr = d_fc2_bias.ptr;
            if (!ok(cublasLtMatmulDescSetAttribute(
                    fc2_op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
                !ok(cublasLtMatmulDescSetAttribute(
                    fc2_op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) ||
                !ok(cublasLtMatmulDescSetAttribute(fc2_op,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &fc2_bias_ptr,
                                                   sizeof(fc2_bias_ptr)))) {
                cleanup();
                return 1;
            }
        }
    }

    if (!ok(cublasLtMatrixLayoutCreate(&fc1_a,
                                       CUDA_R_16F,
                                       static_cast<uint64_t>(input_dim),
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<int64_t>(input_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc1_b,
                                       CUDA_R_16F,
                                       static_cast<uint64_t>(input_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(input_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc1_c,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(hidden_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc2_a,
                                       CUDA_R_16F,
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<uint64_t>(output_dim),
                                       static_cast<int64_t>(hidden_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc2_b,
                                       CUDA_R_16F,
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(hidden_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc2_c,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(output_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(output_dim))) ||
        !ok(cublasLtMatmulPreferenceCreate(&pref))) {
        cleanup();
        return 1;
    }

    const int32_t batch_count = static_cast<int32_t>(batches);
    const int64_t a_stride = 0;
    const int64_t fc1_b_stride = input_dim * cols;
    const int64_t fc1_c_stride = hidden_dim * cols;
    const int64_t fc2_b_stride = hidden_dim * cols;
    const int64_t fc2_c_stride = output_dim * cols;
    for (auto* layout : {fc1_a, fc1_b, fc1_c, fc2_a, fc2_b, fc2_c}) {
        if (!ok(cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)))) {
            cleanup();
            return 1;
        }
    }
    if (!ok(cublasLtMatrixLayoutSetAttribute(
            fc1_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            fc2_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc1_b,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc1_b_stride,
                                             sizeof(fc1_b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc1_c,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc1_c_stride,
                                             sizeof(fc1_c_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc2_b,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc2_b_stride,
                                             sizeof(fc2_b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc2_c,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc2_c_stride,
                                             sizeof(fc2_c_stride))) ||
        !ok(cublasLtMatmulPreferenceSetAttribute(pref,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &options.workspace_bytes,
                                                 sizeof(options.workspace_bytes)))) {
        cleanup();
        return 1;
    }

    constexpr int max_heuristics = 16;
    cublasLtMatmulHeuristicResult_t fc1_heuristics[max_heuristics] = {};
    cublasLtMatmulHeuristicResult_t fc2_heuristics[max_heuristics] = {};
    int fc1_returned = 0;
    int fc2_returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(lt,
                                           fc1_op,
                                           fc1_a,
                                           fc1_b,
                                           fc1_c,
                                           fc1_c,
                                           pref,
                                           max_heuristics,
                                           fc1_heuristics,
                                           &fc1_returned)) ||
        fc1_returned == 0 ||
        !ok(cublasLtMatmulAlgoGetHeuristic(lt,
                                           fc2_op,
                                           fc2_a,
                                           fc2_b,
                                           fc2_c,
                                           fc2_c,
                                           pref,
                                           max_heuristics,
                                           fc2_heuristics,
                                           &fc2_returned)) ||
        fc2_returned == 0) {
        cleanup();
        return 1;
    }
    if (options.fc1_algo_index < 0 || options.fc1_algo_index >= fc1_returned ||
        options.fc2_algo_index < 0 || options.fc2_algo_index >= fc2_returned) {
        cleanup();
        return 3;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        cleanup();
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch = [&] {
        if (!ok(cublasLtMatmul(lt,
                               fc1_op,
                               &alpha,
                               d_fc1_weights_f16.ptr,
                               fc1_a,
                               d_input_f16.ptr,
                               fc1_b,
                               &beta,
                               d_fc1_f32.ptr,
                               fc1_c,
                               d_fc1_f32.ptr,
                               fc1_c,
                               &fc1_heuristics[options.fc1_algo_index].algo,
                               workspace.ptr,
                               options.workspace_bytes,
                               nullptr))) {
            return false;
        }
        gelu_erf_f32_to_f16_kernel<<<static_cast<unsigned>((fc1_elems + convert_block - 1) /
                                                           convert_block),
                                     convert_block>>>(
            static_cast<const float*>(d_fc1_f32.ptr), static_cast<half*>(d_fc1_f16.ptr), fc1_elems);
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }
        return ok(cublasLtMatmul(lt,
                                 fc2_op,
                                 &alpha,
                                 d_fc2_weights_f16.ptr,
                                 fc2_a,
                                 d_fc1_f16.ptr,
                                 fc2_b,
                                 &beta,
                                 d_output_f32.ptr,
                                 fc2_c,
                                 d_output_f32.ptr,
                                 fc2_c,
                                 &fc2_heuristics[options.fc2_algo_index].algo,
                                 workspace.ptr,
                                 options.workspace_bytes,
                                 nullptr));
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cleanup();
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        if (!launch()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    const bool copy_ok = cudaMemcpy(output,
                                    d_output_f32.ptr,
                                    static_cast<size_t>(output_elems) * sizeof(float),
                                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    if (!copy_ok) {
        return 1;
    }

    double checksum = 0.0;
    for (int64_t i = 0; i < output_elems; ++i) {
        checksum += static_cast<double>(output[i]);
    }
    *result = Sam3F16MlpChainResult{
        .chain = summarize(std::move(samples)),
        .output_checksum = checksum,
    };
    return 0;
}

int sam3_bf16_mlp_chain_cublaslt_bench(const float* fc1_weights,
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
                                       Sam3Bf16MlpChainResult* result) {
    if (fc1_weights == nullptr || fc2_weights == nullptr || input == nullptr || output == nullptr ||
        result == nullptr || input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0 || cols <= 0 ||
        batches <= 0 || warmup < 0 || iters <= 0 ||
        input_dim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        hidden_dim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        output_dim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return 2;
    }

    const int64_t ncols_total = cols * batches;
    const int64_t input_elems = input_dim * ncols_total;
    const int64_t fc1_weight_elems = input_dim * hidden_dim;
    const int64_t fc2_weight_elems = hidden_dim * output_dim;
    const int64_t fc1_elems = hidden_dim * ncols_total;
    const int64_t output_elems = output_dim * ncols_total;

    DeviceBuffer d_fc1_weights_f32;
    DeviceBuffer d_fc2_weights_f32;
    DeviceBuffer d_input_f32;
    DeviceBuffer d_fc1_weights_bf16;
    DeviceBuffer d_fc2_weights_bf16;
    DeviceBuffer d_input_bf16;
    DeviceBuffer d_fc1_bias;
    DeviceBuffer d_fc2_bias;
    DeviceBuffer d_fc1_f32;
    DeviceBuffer d_fc1_bf16;
    DeviceBuffer d_output_f32;
    DeviceBuffer workspace;

    if (!alloc(d_fc1_weights_f32, static_cast<size_t>(fc1_weight_elems) * sizeof(float)) ||
        !alloc(d_fc2_weights_f32, static_cast<size_t>(fc2_weight_elems) * sizeof(float)) ||
        !alloc(d_input_f32, static_cast<size_t>(input_elems) * sizeof(float)) ||
        !alloc(d_fc1_weights_bf16, static_cast<size_t>(fc1_weight_elems) * sizeof(nv_bfloat16)) ||
        !alloc(d_fc2_weights_bf16, static_cast<size_t>(fc2_weight_elems) * sizeof(nv_bfloat16)) ||
        !alloc(d_input_bf16, static_cast<size_t>(input_elems) * sizeof(nv_bfloat16)) ||
        (fc1_bias != nullptr &&
         !alloc(d_fc1_bias, static_cast<size_t>(hidden_dim) * sizeof(float))) ||
        (fc2_bias != nullptr &&
         !alloc(d_fc2_bias, static_cast<size_t>(output_dim) * sizeof(float))) ||
        !alloc(d_fc1_f32, static_cast<size_t>(fc1_elems) * sizeof(float)) ||
        !alloc(d_fc1_bf16, static_cast<size_t>(fc1_elems) * sizeof(nv_bfloat16)) ||
        !alloc(d_output_f32, static_cast<size_t>(output_elems) * sizeof(float)) ||
        !alloc(workspace, workspace_bytes)) {
        return 1;
    }

    if (cudaMemcpy(d_fc1_weights_f32.ptr,
                   fc1_weights,
                   static_cast<size_t>(fc1_weight_elems) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_fc2_weights_f32.ptr,
                   fc2_weights,
                   static_cast<size_t>(fc2_weight_elems) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_input_f32.ptr,
                   input,
                   static_cast<size_t>(input_elems) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (fc1_bias != nullptr && cudaMemcpy(d_fc1_bias.ptr,
                                           fc1_bias,
                                           static_cast<size_t>(hidden_dim) * sizeof(float),
                                           cudaMemcpyHostToDevice) != cudaSuccess) ||
        (fc2_bias != nullptr && cudaMemcpy(d_fc2_bias.ptr,
                                           fc2_bias,
                                           static_cast<size_t>(output_dim) * sizeof(float),
                                           cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    constexpr int convert_block = 256;
    f32_to_bf16_kernel<<<static_cast<unsigned>((fc1_weight_elems + convert_block - 1) /
                                               convert_block),
                         convert_block>>>(static_cast<const float*>(d_fc1_weights_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_fc1_weights_bf16.ptr),
                                          fc1_weight_elems);
    f32_to_bf16_kernel<<<static_cast<unsigned>((fc2_weight_elems + convert_block - 1) /
                                               convert_block),
                         convert_block>>>(static_cast<const float*>(d_fc2_weights_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_fc2_weights_bf16.ptr),
                                          fc2_weight_elems);
    f32_to_bf16_kernel<<<static_cast<unsigned>((input_elems + convert_block - 1) / convert_block),
                         convert_block>>>(static_cast<const float*>(d_input_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_input_bf16.ptr),
                                          input_elems);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t fc1_op = nullptr;
    cublasLtMatmulDesc_t fc2_op = nullptr;
    cublasLtMatrixLayout_t fc1_a = nullptr;
    cublasLtMatrixLayout_t fc1_b = nullptr;
    cublasLtMatrixLayout_t fc1_c = nullptr;
    cublasLtMatrixLayout_t fc2_a = nullptr;
    cublasLtMatrixLayout_t fc2_b = nullptr;
    cublasLtMatrixLayout_t fc2_c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&] {
        if (pref != nullptr) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        for (auto* layout : {fc2_c, fc2_b, fc2_a, fc1_c, fc1_b, fc1_a}) {
            if (layout != nullptr) {
                cublasLtMatrixLayoutDestroy(layout);
            }
        }
        if (fc2_op != nullptr) {
            cublasLtMatmulDescDestroy(fc2_op);
        }
        if (fc1_op != nullptr) {
            cublasLtMatmulDescDestroy(fc1_op);
        }
        if (lt != nullptr) {
            cublasLtDestroy(lt);
        }
    };

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    const cublasComputeType_t compute_type = compute == Sam3Bf16CublasLtCompute::fast_16bf
                                                 ? CUBLAS_COMPUTE_32F_FAST_16BF
                                                 : CUBLAS_COMPUTE_32F;
    if (!ok(cublasLtCreate(&lt)) ||
        !ok(cublasLtMatmulDescCreate(&fc1_op, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatmulDescCreate(&fc2_op, compute_type, CUDA_R_32F)) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc1_op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc1_op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc2_op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))) ||
        !ok(cublasLtMatmulDescSetAttribute(
            fc2_op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)))) {
        cleanup();
        return 1;
    }

    if ((fc1_bias != nullptr && !direct_fc1_bf16_no_bias) || fc2_bias != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const cudaDataType_t bias_type = CUDA_R_32F;
        if (fc1_bias != nullptr && !direct_fc1_bf16_no_bias) {
            const void* fc1_bias_ptr = d_fc1_bias.ptr;
            if (!ok(cublasLtMatmulDescSetAttribute(
                    fc1_op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
                !ok(cublasLtMatmulDescSetAttribute(
                    fc1_op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) ||
                !ok(cublasLtMatmulDescSetAttribute(fc1_op,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &fc1_bias_ptr,
                                                   sizeof(fc1_bias_ptr)))) {
                cleanup();
                return 1;
            }
        }
        if (fc2_bias != nullptr) {
            const void* fc2_bias_ptr = d_fc2_bias.ptr;
            if (!ok(cublasLtMatmulDescSetAttribute(
                    fc2_op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) ||
                !ok(cublasLtMatmulDescSetAttribute(
                    fc2_op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type))) ||
                !ok(cublasLtMatmulDescSetAttribute(fc2_op,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &fc2_bias_ptr,
                                                   sizeof(fc2_bias_ptr)))) {
                cleanup();
                return 1;
            }
        }
    }

    if (!ok(cublasLtMatrixLayoutCreate(&fc1_a,
                                       CUDA_R_16BF,
                                       static_cast<uint64_t>(input_dim),
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<int64_t>(input_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc1_b,
                                       CUDA_R_16BF,
                                       static_cast<uint64_t>(input_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(input_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc1_c,
                                       direct_fc1_bf16_no_bias ? CUDA_R_16BF : CUDA_R_32F,
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(hidden_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc2_a,
                                       CUDA_R_16BF,
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<uint64_t>(output_dim),
                                       static_cast<int64_t>(hidden_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc2_b,
                                       CUDA_R_16BF,
                                       static_cast<uint64_t>(hidden_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(hidden_dim))) ||
        !ok(cublasLtMatrixLayoutCreate(&fc2_c,
                                       CUDA_R_32F,
                                       static_cast<uint64_t>(output_dim),
                                       static_cast<uint64_t>(cols),
                                       static_cast<int64_t>(output_dim))) ||
        !ok(cublasLtMatmulPreferenceCreate(&pref))) {
        cleanup();
        return 1;
    }

    const int32_t batch_count = static_cast<int32_t>(batches);
    const int64_t a_stride = 0;
    const int64_t fc1_b_stride = input_dim * cols;
    const int64_t fc1_c_stride = hidden_dim * cols;
    const int64_t fc2_b_stride = hidden_dim * cols;
    const int64_t fc2_c_stride = output_dim * cols;
    for (auto* layout : {fc1_a, fc1_b, fc1_c, fc2_a, fc2_b, fc2_c}) {
        if (!ok(cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)))) {
            cleanup();
            return 1;
        }
    }
    if (!ok(cublasLtMatrixLayoutSetAttribute(
            fc1_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(
            fc2_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc1_b,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc1_b_stride,
                                             sizeof(fc1_b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc1_c,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc1_c_stride,
                                             sizeof(fc1_c_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc2_b,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc2_b_stride,
                                             sizeof(fc2_b_stride))) ||
        !ok(cublasLtMatrixLayoutSetAttribute(fc2_c,
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &fc2_c_stride,
                                             sizeof(fc2_c_stride))) ||
        !ok(cublasLtMatmulPreferenceSetAttribute(pref,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_bytes,
                                                 sizeof(workspace_bytes)))) {
        cleanup();
        return 1;
    }

    cublasLtMatmulHeuristicResult_t fc1_heuristic{};
    cublasLtMatmulHeuristicResult_t fc2_heuristic{};
    int returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(
            lt, fc1_op, fc1_a, fc1_b, fc1_c, fc1_c, pref, 1, &fc1_heuristic, &returned)) ||
        returned == 0 ||
        !ok(cublasLtMatmulAlgoGetHeuristic(
            lt, fc2_op, fc2_a, fc2_b, fc2_c, fc2_c, pref, 1, &fc2_heuristic, &returned)) ||
        returned == 0) {
        cleanup();
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        cleanup();
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch = [&] {
        void* fc1_dst = direct_fc1_bf16_no_bias ? d_fc1_bf16.ptr : d_fc1_f32.ptr;
        if (!ok(cublasLtMatmul(lt,
                               fc1_op,
                               &alpha,
                               d_fc1_weights_bf16.ptr,
                               fc1_a,
                               d_input_bf16.ptr,
                               fc1_b,
                               &beta,
                               fc1_dst,
                               fc1_c,
                               fc1_dst,
                               fc1_c,
                               &fc1_heuristic.algo,
                               workspace.ptr,
                               workspace_bytes,
                               nullptr))) {
            return false;
        }
        if (direct_fc1_bf16_no_bias) {
            add_bias_gelu_erf_bf16_inplace_kernel<<<
                static_cast<unsigned>((fc1_elems + convert_block - 1) / convert_block),
                convert_block>>>(static_cast<nv_bfloat16*>(d_fc1_bf16.ptr),
                                 static_cast<const float*>(d_fc1_bias.ptr),
                                 static_cast<int>(hidden_dim),
                                 fc1_elems);
        } else {
            gelu_erf_f32_to_bf16_kernel<<<static_cast<unsigned>((fc1_elems + convert_block - 1) /
                                                                convert_block),
                                          convert_block>>>(
                static_cast<const float*>(d_fc1_f32.ptr),
                static_cast<nv_bfloat16*>(d_fc1_bf16.ptr),
                fc1_elems);
        }
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }
        return ok(cublasLtMatmul(lt,
                                 fc2_op,
                                 &alpha,
                                 d_fc2_weights_bf16.ptr,
                                 fc2_a,
                                 d_fc1_bf16.ptr,
                                 fc2_b,
                                 &beta,
                                 d_output_f32.ptr,
                                 fc2_c,
                                 d_output_f32.ptr,
                                 fc2_c,
                                 &fc2_heuristic.algo,
                                 workspace.ptr,
                                 workspace_bytes,
                                 nullptr));
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cleanup();
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        if (!launch()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cleanup();
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    const size_t output_bytes = static_cast<size_t>(output_elems) * sizeof(float);
    const bool copy_ok =
        cudaMemcpy(output, d_output_f32.ptr, output_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    if (!copy_ok) {
        return 1;
    }

    double checksum = 0.0;
    const int64_t stride = std::max<int64_t>(1, output_elems / 4096);
    for (int64_t i = 0; i < output_elems; i += stride) {
        checksum += static_cast<double>(output[i]);
    }

    *result = Sam3Bf16MlpChainResult{
        .chain = summarize(std::move(samples)),
        .output_checksum = checksum,
    };
    return 0;
}

int sam3_bf16_wmma_bench(const float* weights,
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
                         Sam3F32BatchedCustomResult* result) {
    if (weights == nullptr || input == nullptr || output == nullptr || result == nullptr ||
        k <= 0 || rows <= 0 || cols <= 0 || batches <= 0 || warmup < 0 || iters <= 0 ||
        k > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        batches > static_cast<int64_t>(std::numeric_limits<int>::max()) || k % kTile != 0 ||
        rows % kTile != 0 || cols % kTile != 0) {
        return 2;
    }

    const int64_t input_elems = k * cols * batches;
    const int64_t weight_elems = k * rows;
    const int64_t output_elems = rows * cols * batches;
    const size_t weight_f32_bytes = static_cast<size_t>(weight_elems) * sizeof(float);
    const size_t input_f32_bytes = static_cast<size_t>(input_elems) * sizeof(float);
    const size_t weight_bf16_bytes = static_cast<size_t>(weight_elems) * sizeof(nv_bfloat16);
    const size_t input_bf16_bytes = static_cast<size_t>(input_elems) * sizeof(nv_bfloat16);
    const size_t bias_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t output_f32_bytes = static_cast<size_t>(output_elems) * sizeof(float);
    const size_t output_bf16_bytes = static_cast<size_t>(output_elems) * sizeof(nv_bfloat16);

    DeviceBuffer d_weights_f32;
    DeviceBuffer d_input_f32;
    DeviceBuffer d_weights_bf16;
    DeviceBuffer d_input_bf16;
    DeviceBuffer d_bias;
    DeviceBuffer d_output_f32;
    DeviceBuffer d_output_bf16;
    if (!alloc(d_weights_f32, weight_f32_bytes) || !alloc(d_input_f32, input_f32_bytes) ||
        !alloc(d_weights_bf16, weight_bf16_bytes) || !alloc(d_input_bf16, input_bf16_bytes) ||
        (gelu_bf16_output ? !alloc(d_output_bf16, output_bf16_bytes)
                          : !alloc(d_output_f32, output_f32_bytes)) ||
        (gelu_bf16_output && !alloc(d_output_f32, output_f32_bytes)) ||
        (bias != nullptr && !alloc(d_bias, bias_bytes))) {
        return 1;
    }

    if (cudaMemcpy(d_weights_f32.ptr, weights, weight_f32_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_input_f32.ptr, input, input_f32_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        (bias != nullptr &&
         cudaMemcpy(d_bias.ptr, bias, bias_bytes, cudaMemcpyHostToDevice) != cudaSuccess)) {
        return 1;
    }

    constexpr int convert_block = 256;
    f32_to_bf16_kernel<<<static_cast<unsigned>((weight_elems + convert_block - 1) / convert_block),
                         convert_block>>>(static_cast<const float*>(d_weights_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_weights_bf16.ptr),
                                          weight_elems);
    f32_to_bf16_kernel<<<static_cast<unsigned>((input_elems + convert_block - 1) / convert_block),
                         convert_block>>>(static_cast<const float*>(d_input_f32.ptr),
                                          static_cast<nv_bfloat16*>(d_input_bf16.ptr),
                                          input_elems);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        return 1;
    }

    const dim3 block(32, 1, 1);
    const dim3 grid(static_cast<unsigned>(cols / kTile),
                    static_cast<unsigned>(rows / kTile),
                    static_cast<unsigned>(batches));
    auto launch = [&] {
        if (gelu_bf16_output) {
            bf16_wmma_matmul_bias_gelu_bf16_kernel<<<grid, block>>>(
                static_cast<const nv_bfloat16*>(d_weights_bf16.ptr),
                static_cast<const nv_bfloat16*>(d_input_bf16.ptr),
                static_cast<const float*>(d_bias.ptr),
                static_cast<nv_bfloat16*>(d_output_bf16.ptr),
                static_cast<int>(k),
                static_cast<int>(rows),
                static_cast<int>(cols));
        } else {
            bf16_wmma_matmul_bias_kernel<<<grid, block>>>(
                static_cast<const nv_bfloat16*>(d_weights_bf16.ptr),
                static_cast<const nv_bfloat16*>(d_input_bf16.ptr),
                static_cast<const float*>(d_bias.ptr),
                static_cast<float*>(d_output_f32.ptr),
                static_cast<int>(k),
                static_cast<int>(rows),
                static_cast<int>(cols));
        }
        return cudaGetLastError();
    };

    for (int i = 0; i < warmup; ++i) {
        if (launch() != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        if (launch() != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        samples.push_back(elapsed_ms(start, stop));
    }

    if (gelu_bf16_output) {
        constexpr int convert_block = 256;
        bf16_to_f32_kernel<<<static_cast<unsigned>((output_elems + convert_block - 1) /
                                                   convert_block),
                             convert_block>>>(static_cast<const nv_bfloat16*>(d_output_bf16.ptr),
                                              static_cast<float*>(d_output_f32.ptr),
                                              output_elems);
    }
    const bool copy_ok =
        cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, d_output_f32.ptr, output_f32_bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!copy_ok) {
        return 1;
    }

    *result = summarize(std::move(samples));
    return 0;
}
