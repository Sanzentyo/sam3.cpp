#include "../ggml/src/ggml-cuda/quantize.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

struct Args {
    int64_t k = 448;
    int64_t cols = 32768;
    int64_t padded_k = 512;
    int cols_per_block = 4;
    int warmup = 5;
    int iters = 50;
    bool show_help = false;
};

static void usage(const char * argv0) {
    std::cerr << "Usage: " << argv0
              << " [--k <n>] [--cols <n>] [--padded-k <n>] [--cols-per-block 4|8|16]"
              << " [--warmup <n>] [--iters <n>]\n";
}

static bool parse_int64(const char * s, int64_t & out) {
    const std::string_view sv{s};
    int64_t v = 0;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || ptr != sv.data() + sv.size() || v <= 0) {
        return false;
    }
    out = v;
    return true;
}

static bool parse_int(const char * s, int & out) {
    int64_t v = 0;
    if (!parse_int64(s, v) || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_nonnegative_int(const char * s, int & out) {
    const std::string_view sv{s};
    int64_t v = 0;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || ptr != sv.data() + sv.size() || v < 0 ||
        v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
            return true;
        }
        if (arg == "--k") {
            const char * v = need_value("--k");
            if (v == nullptr || !parse_int64(v, args.k)) {
                return false;
            }
        } else if (arg == "--cols") {
            const char * v = need_value("--cols");
            if (v == nullptr || !parse_int64(v, args.cols)) {
                return false;
            }
        } else if (arg == "--padded-k") {
            const char * v = need_value("--padded-k");
            if (v == nullptr || !parse_int64(v, args.padded_k)) {
                return false;
            }
        } else if (arg == "--cols-per-block") {
            const char * v = need_value("--cols-per-block");
            if (v == nullptr || !parse_int(v, args.cols_per_block)) {
                return false;
            }
        } else if (arg == "--warmup") {
            const char * v = need_value("--warmup");
            if (v == nullptr || !parse_nonnegative_int(v, args.warmup)) {
                return false;
            }
        } else if (arg == "--iters") {
            const char * v = need_value("--iters");
            if (v == nullptr || !parse_int(v, args.iters)) {
                return false;
            }
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

static float deterministic_value(size_t i) {
    uint32_t x = static_cast<uint32_t>(i * 1664525u + 1013904223u);
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    const float u = static_cast<float>(x & 0xffffu) / 65535.0f;
    return (u - 0.5f) * 0.25f;
}

static float synthetic_q4_1_value(size_t row, size_t k) {
    const uint32_t seed = static_cast<uint32_t>(row * 1315423911u + k * 2654435761u);
    const float d = 0.0005f + static_cast<float>((seed >> 8) & 0xffu) / 255.0f * 0.004f;
    const float m = (static_cast<float>((seed >> 20) & 0xffu) / 255.0f - 0.5f) * 0.03f;
    const float q = static_cast<float>(seed & 0x0fu);
    return m + d * q;
}

template <int cols_per_block>
__global__ void quantize_q4_1_ds4_candidate(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int64_t ne1) {
    constexpr int vals_per_mmq_block = 4 * QK8_1;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int64_t i1 = static_cast<int64_t>(blockIdx.x) * cols_per_block + warp;
    if (i1 >= ne1) {
        return;
    }

    const int64_t qblock = blockIdx.y;
    const int64_t i0 = qblock * vals_per_mmq_block + lane * 4;
    const int64_t base = i1 * s01 + i0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4 xi = i0 < ne00 ? x4[base / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    const float d_inv = 127.0f / amax;
    const char4 q = make_char4(roundf(xi.x * d_inv),
                               roundf(xi.y * d_inv),
                               roundf(xi.z * d_inv),
                               roundf(xi.w * d_inv));

    block_q8_1_mmq * y = static_cast<block_q8_1_mmq *>(vy);
    const int64_t ib = qblock * ne1 + i1;
    char4 * yqs4 = reinterpret_cast<char4 *>(y[ib].qs);
    yqs4[lane] = q;

    const int iqs = lane * 4;
    if (iqs % 32 == 0) {
        const float d = 1.0f / d_inv;
        y[ib].ds4[iqs / 32] = make_half2(d, sum);
    }

    GGML_UNUSED(ne0);
}

static bool cuda_ok(cudaError_t err, std::string_view what) {
    if (err == cudaSuccess) {
        return true;
    }
    std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
    return false;
}

static void launch_candidate(const Args & args, const float * x, void * out, cudaStream_t stream) {
    const int64_t qblocks = args.padded_k / (4 * QK8_1);
    if (args.cols_per_block == 16) {
        quantize_q4_1_ds4_candidate<16>
            <<<dim3((args.cols + 15) / 16, qblocks, 1), dim3(WARP_SIZE, 16, 1), 0, stream>>>(
                x, out, args.k, args.k, args.padded_k, args.cols);
    } else if (args.cols_per_block == 8) {
        quantize_q4_1_ds4_candidate<8>
            <<<dim3((args.cols + 7) / 8, qblocks, 1), dim3(WARP_SIZE, 8, 1), 0, stream>>>(
                x, out, args.k, args.k, args.padded_k, args.cols);
    } else {
        quantize_q4_1_ds4_candidate<4>
            <<<dim3((args.cols + 3) / 4, qblocks, 1), dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                x, out, args.k, args.k, args.padded_k, args.cols);
    }
}

static bool time_quantizers(
        const Args & args, const float * x, void * ref_out, void * cand_out, cudaStream_t stream,
        float & ref_ms, float & cand_ms) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "cudaEventCreate start") ||
        !cuda_ok(cudaEventCreate(&stop), "cudaEventCreate stop")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }

    for (int i = 0; i < args.warmup; ++i) {
        quantize_mmq_q8_1_cuda(x, nullptr, ref_out, GGML_TYPE_Q4_1, args.k, args.k, 0, 0,
                               args.padded_k, args.cols, 1, 1, stream);
        launch_candidate(args, x, cand_out, stream);
    }
    if (!cuda_ok(cudaGetLastError(), "warmup kernel launch") ||
        !cuda_ok(cudaStreamSynchronize(stream), "warmup synchronize")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return false;
    }

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        quantize_mmq_q8_1_cuda(x, nullptr, ref_out, GGML_TYPE_Q4_1, args.k, args.k, 0, 0,
                               args.padded_k, args.cols, 1, 1, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    if (!cuda_ok(cudaEventSynchronize(stop), "reference timing synchronize")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return false;
    }
    float ref_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ref_total, start, stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        launch_candidate(args, x, cand_out, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    if (!cuda_ok(cudaEventSynchronize(stop), "candidate timing synchronize")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return false;
    }
    float cand_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cand_total, start, stop));

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    ref_ms = ref_total / static_cast<float>(args.iters);
    cand_ms = cand_total / static_cast<float>(args.iters);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    if (args.show_help) {
        usage(argv[0]);
        return 0;
    }
    if (args.cols_per_block != 4 && args.cols_per_block != 8 && args.cols_per_block != 16) {
        std::cerr << "--cols-per-block must be one of 4, 8, or 16\n";
        return 2;
    }
    if (args.k % 4 != 0 || args.padded_k % (4 * QK8_1) != 0 || args.padded_k < args.k) {
        std::cerr << "--k must be divisible by 4 and --padded-k must be a >=k multiple of 128\n";
        return 2;
    }
    if (args.warmup < 0 || args.iters <= 0) {
        std::cerr << "--warmup must be non-negative and --iters must be positive\n";
        return 2;
    }

    setenv("GGML_CUDA_MMQ_Q8_1_WARP_COLS", std::to_string(args.cols_per_block).c_str(), 1);

    const size_t input_elems = static_cast<size_t>(args.k * args.cols);
    std::vector<float> input(input_elems);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = deterministic_value(i);
    }

    const int64_t qblocks = args.padded_k / (4 * QK8_1);
    const size_t output_bytes = static_cast<size_t>(qblocks * args.cols) * sizeof(block_q8_1_mmq);

    float * x_d = nullptr;
    void * ref_d = nullptr;
    void * cand_d = nullptr;
    cudaStream_t stream = nullptr;

    bool ok = cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") &&
              cuda_ok(cudaMalloc(&x_d, input.size() * sizeof(float)), "cudaMalloc input") &&
              cuda_ok(cudaMalloc(&ref_d, output_bytes), "cudaMalloc ref") &&
              cuda_ok(cudaMalloc(&cand_d, output_bytes), "cudaMalloc candidate") &&
              cuda_ok(cudaMemcpyAsync(x_d, input.data(), input.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync input") &&
              cuda_ok(cudaMemsetAsync(ref_d, 0, output_bytes, stream), "cudaMemsetAsync ref") &&
              cuda_ok(cudaMemsetAsync(cand_d, 0, output_bytes, stream), "cudaMemsetAsync candidate");
    if (ok) {
        quantize_mmq_q8_1_cuda(x_d, nullptr, ref_d, GGML_TYPE_Q4_1, args.k, args.k, 0, 0,
                               args.padded_k, args.cols, 1, 1, stream);
        launch_candidate(args, x_d, cand_d, stream);
        ok = cuda_ok(cudaGetLastError(), "kernel launch") &&
             cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    std::vector<uint8_t> ref(output_bytes);
    std::vector<uint8_t> cand(output_bytes);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(ref.data(), ref_d, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy ref") &&
             cuda_ok(cudaMemcpy(cand.data(), cand_d, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy candidate");
    }

    size_t mismatch = 0;
    size_t first = 0;
    size_t qs_mismatch = 0;
    size_t ds_mismatch = 0;
    size_t first_qs = 0;
    size_t first_ds = 0;
    int max_qs_abs_diff = 0;
    size_t qs_abs_diff_gt1 = 0;
    float max_sampled_dot_abs_diff = 0.0f;
    float ref_ms = 0.0f;
    float cand_ms = 0.0f;
    if (ok) {
        for (size_t i = 0; i < output_bytes; ++i) {
            if (ref[i] != cand[i]) {
                if (mismatch == 0) {
                    first = i;
                }
                ++mismatch;
            }
        }

        const auto * ref_blocks = reinterpret_cast<const block_q8_1_mmq *>(ref.data());
        const auto * cand_blocks = reinterpret_cast<const block_q8_1_mmq *>(cand.data());
        const size_t nblocks = static_cast<size_t>(qblocks * args.cols);
        for (size_t block = 0; block < nblocks; ++block) {
            bool block_differs = false;
            const auto * ref_ds = reinterpret_cast<const uint8_t *>(ref_blocks[block].ds4);
            const auto * cand_ds = reinterpret_cast<const uint8_t *>(cand_blocks[block].ds4);
            for (size_t i = 0; i < sizeof(ref_blocks[block].ds4); ++i) {
                if (ref_ds[i] != cand_ds[i]) {
                    if (ds_mismatch == 0) {
                        first_ds = block * sizeof(block_q8_1_mmq) + i;
                    }
                    ++ds_mismatch;
                    block_differs = true;
                }
            }
            for (size_t i = 0; i < sizeof(ref_blocks[block].qs); ++i) {
                if (ref_blocks[block].qs[i] != cand_blocks[block].qs[i]) {
                    if (qs_mismatch == 0) {
                        first_qs = block * sizeof(block_q8_1_mmq) + sizeof(ref_blocks[block].ds4) + i;
                    }
                    ++qs_mismatch;
                    const int diff = std::abs(static_cast<int>(ref_blocks[block].qs[i]) -
                                              static_cast<int>(cand_blocks[block].qs[i]));
                    max_qs_abs_diff = std::max(max_qs_abs_diff, diff);
                    if (diff > 1) {
                        ++qs_abs_diff_gt1;
                    }
                    block_differs = true;
                }
            }
            if (block_differs) {
                const size_t qblock = block / static_cast<size_t>(args.cols);
                for (size_t row = 0; row < 112; ++row) {
                    float dot_diff = 0.0f;
                    for (size_t i = 0; i < sizeof(ref_blocks[block].qs); ++i) {
                        const size_t k = qblock * 4 * QK8_1 + i;
                        if (k >= static_cast<size_t>(args.k)) {
                            break;
                        }
                        const float2 ref_ds_i = __half22float2(ref_blocks[block].ds4[i / 32]);
                        const float2 cand_ds_i = __half22float2(cand_blocks[block].ds4[i / 32]);
                        dot_diff += synthetic_q4_1_value(row, k) *
                                    (static_cast<float>(ref_blocks[block].qs[i]) * ref_ds_i.x -
                                     static_cast<float>(cand_blocks[block].qs[i]) * cand_ds_i.x);
                    }
                    max_sampled_dot_abs_diff = std::max(max_sampled_dot_abs_diff, std::abs(dot_diff));
                }
            }
        }
        ok = time_quantizers(args, x_d, ref_d, cand_d, stream, ref_ms, cand_ms);
    }

    cudaFree(cand_d);
    cudaFree(ref_d);
    cudaFree(x_d);
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }

    if (!ok) {
        return 1;
    }

    std::cout << "q8_layout_parity type=q4_1"
              << " k=" << args.k
              << " padded_k=" << args.padded_k
              << " cols=" << args.cols
              << " cols_per_block=" << args.cols_per_block
              << " warmup=" << args.warmup
              << " iters=" << args.iters
              << " bytes=" << output_bytes
              << " mismatches=" << mismatch
              << " first_mismatch=" << (mismatch == 0 ? 0 : first)
              << " qs_mismatches=" << qs_mismatch
              << " first_qs_mismatch=" << (qs_mismatch == 0 ? 0 : first_qs)
              << " max_qs_abs_diff=" << max_qs_abs_diff
              << " qs_abs_diff_gt1=" << qs_abs_diff_gt1
              << " max_sampled_dot_abs_diff=" << max_sampled_dot_abs_diff
              << " ref_ms=" << ref_ms
              << " cand_ms=" << cand_ms
              << " speedup=" << (cand_ms > 0.0f ? ref_ms / cand_ms : 0.0f)
              << " ds_mismatches=" << ds_mismatch
              << " first_ds_mismatch=" << (ds_mismatch == 0 ? 0 : first_ds)
              << "\n";
    return qs_abs_diff_gt1 == 0 ? 0 : 1;
}
