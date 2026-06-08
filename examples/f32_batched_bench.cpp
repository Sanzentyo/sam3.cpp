#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "f32_batched_custom.cuh"
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

namespace {

struct Args {
    int64_t k = 112;
    int64_t rows = 336;
    int64_t cols = 64;
    int64_t batches = 1024;
    int warmup = 3;
    int iters = 20;
    int threads = 4;
    float tolerance = 5e-5f;
    int64_t check_samples = 4096;
    bool cuda = true;
    bool bias = true;
    bool check = false;
    bool custom_cuda = false;
    bool cublaslt_bias = false;
    bool cublaslt_fast_tf32 = false;
    bool bf16_cublaslt = false;
    bool bf16_direct_dst = false;
    bool bf16_fast_compute = false;
    bool gelu_epilogue = false;
    bool gelu_quant_ds4 = false;
    bool show_help = false;
};

struct BackendDeleter {
    void operator()(ggml_backend* backend) const {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct ContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GallocrDeleter {
    void operator()(ggml_gallocr* alloc) const {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using ContextPtr = std::unique_ptr<ggml_context, ContextDeleter>;
using GallocrPtr = std::unique_ptr<ggml_gallocr, GallocrDeleter>;

struct BenchResult {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

static void usage(const char* argv0) {
    std::cerr << std::format(
        "Usage: {} [--cpu|--cuda] [--k <n>] [--rows <n>] [--cols <n>] [--batches <n>] "
        "[--warmup <n>] [--iters <n>] [--threads <n>] [--bias|--no-bias] [--check] "
        "[--check-samples <n>] [--tolerance <f>] "
        "[--custom-cuda|--cublaslt-bias|--cublaslt-bias-tf32|--bf16-cublaslt|"
        "--gelu-quant-ds4] [--bf16-direct-dst] [--bf16-fast-compute] [--gelu-epilogue]\n",
        argv0);
}

static bool parse_int64(const char* s, int64_t& out) {
    const std::string_view sv{s};
    int64_t v = 0;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || ptr != sv.data() + sv.size() || v <= 0) {
        return false;
    }
    out = v;
    return true;
}

static bool parse_int(const char* s, int& out) {
    int64_t v = 0;
    if (!parse_int64(s, v) || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_float(const char* s, float& out) {
    const std::string_view sv{s};
    float v = 0.0f;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || ptr != sv.data() + sv.size() || !std::isfinite(v) || v <= 0.0f) {
        return false;
    }
    out = v;
    return true;
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << std::format("missing value for {}\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--cpu") {
            args.cuda = false;
        } else if (arg == "--cuda") {
            args.cuda = true;
        } else if (arg == "--custom-cuda") {
            args.custom_cuda = true;
            args.cuda = true;
        } else if (arg == "--cublaslt-bias") {
            args.cublaslt_bias = true;
            args.cuda = true;
        } else if (arg == "--cublaslt-bias-tf32") {
            args.cublaslt_bias = true;
            args.cublaslt_fast_tf32 = true;
            args.cuda = true;
        } else if (arg == "--bf16-cublaslt") {
            args.bf16_cublaslt = true;
            args.cuda = true;
        } else if (arg == "--bf16-direct-dst") {
            args.bf16_direct_dst = true;
        } else if (arg == "--bf16-fast-compute") {
            args.bf16_fast_compute = true;
        } else if (arg == "--gelu-epilogue") {
            args.gelu_epilogue = true;
        } else if (arg == "--gelu-quant-ds4") {
            args.gelu_quant_ds4 = true;
            args.cuda = true;
        } else if (arg == "--bias") {
            args.bias = true;
        } else if (arg == "--no-bias") {
            args.bias = false;
        } else if (arg == "--check") {
            args.check = true;
        } else if (arg == "--k") {
            const char* v = need_value("--k");
            if (v == nullptr || !parse_int64(v, args.k)) {
                return false;
            }
        } else if (arg == "--rows") {
            const char* v = need_value("--rows");
            if (v == nullptr || !parse_int64(v, args.rows)) {
                return false;
            }
        } else if (arg == "--cols") {
            const char* v = need_value("--cols");
            if (v == nullptr || !parse_int64(v, args.cols)) {
                return false;
            }
        } else if (arg == "--batches") {
            const char* v = need_value("--batches");
            if (v == nullptr || !parse_int64(v, args.batches)) {
                return false;
            }
        } else if (arg == "--warmup") {
            const char* v = need_value("--warmup");
            if (v == nullptr || !parse_int(v, args.warmup)) {
                return false;
            }
        } else if (arg == "--iters") {
            const char* v = need_value("--iters");
            if (v == nullptr || !parse_int(v, args.iters)) {
                return false;
            }
        } else if (arg == "--threads") {
            const char* v = need_value("--threads");
            if (v == nullptr || !parse_int(v, args.threads)) {
                return false;
            }
        } else if (arg == "--check-samples") {
            const char* v = need_value("--check-samples");
            if (v == nullptr || !parse_int64(v, args.check_samples)) {
                return false;
            }
        } else if (arg == "--tolerance") {
            const char* v = need_value("--tolerance");
            if (v == nullptr || !parse_float(v, args.tolerance)) {
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            args.show_help = true;
            return true;
        } else {
            std::cerr << std::format("unknown argument: {}\n", argv[i]);
            return false;
        }
    }
    return true;
}

static float deterministic_value(size_t i, int salt) {
    uint32_t x = static_cast<uint32_t>(i * 1664525u + 1013904223u + salt * 747796405u);
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    const float u = static_cast<float>(x & 0xffffu) / 65535.0f;
    return (u - 0.5f) * 0.125f;
}

static std::vector<float> make_input(size_t n, int salt) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = deterministic_value(i, salt);
    }
    return out;
}

static BackendPtr create_backend(const Args& args) {
#ifdef GGML_USE_CUDA
    if (args.cuda) {
        return BackendPtr{ggml_backend_cuda_init(0)};
    }
#else
    if (args.cuda) {
        std::cerr << "CUDA backend is not compiled in\n";
        return nullptr;
    }
#endif
    BackendPtr backend{ggml_backend_cpu_init()};
    if (backend != nullptr) {
        ggml_backend_cpu_set_n_threads(backend.get(), args.threads);
    }
    return backend;
}

static ContextPtr make_context() {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return ContextPtr{ggml_init(params)};
}

static BenchResult summarize(std::vector<double> samples) {
    std::ranges::sort(samples);
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const auto at_quantile = [&](double q) {
        const size_t idx =
            std::min(samples.size() - 1, static_cast<size_t>(std::ceil(q * samples.size())) - 1);
        return samples[idx];
    };

    return BenchResult{
        .mean_ms = sum / static_cast<double>(samples.size()),
        .median_ms = at_quantile(0.50),
        .p95_ms = at_quantile(0.95),
        .min_ms = samples.front(),
        .max_ms = samples.back(),
    };
}

static double run_once_ms(ggml_backend_t backend, ggml_cgraph* graph) {
    using Clock = std::chrono::steady_clock;
    ggml_backend_synchronize(backend);
    const auto t0 = Clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const auto t1 = Clock::now();
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << std::format("graph compute failed: {}\n", ggml_status_to_string(status));
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool check_reference(const Args& args,
                            std::span<const float> weights,
                            std::span<const float> input,
                            std::span<const float> bias,
                            std::span<const float> got) {
    const size_t total = static_cast<size_t>(args.rows * args.cols * args.batches);
    const size_t samples = static_cast<size_t>(std::min<int64_t>(args.check_samples, total));
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    size_t max_i = 0;
    size_t bad = 0;

    const uint64_t stride = std::max<uint64_t>(1, total / samples);
    for (size_t sample = 0, out_i = 0; sample < samples;
         ++sample, out_i = (out_i + stride) % total) {
        const int64_t row = static_cast<int64_t>(out_i % static_cast<size_t>(args.rows));
        const int64_t col = static_cast<int64_t>((out_i / static_cast<size_t>(args.rows)) %
                                                 static_cast<size_t>(args.cols));
        const int64_t batch =
            static_cast<int64_t>(out_i / static_cast<size_t>(args.rows * args.cols));

        float acc = 0.0f;
        for (int64_t kk = 0; kk < args.k; ++kk) {
            acc += weights[static_cast<size_t>(row * args.k + kk)] *
                   input[static_cast<size_t>((batch * args.cols + col) * args.k + kk)];
        }
        if (args.bias) {
            acc += bias[static_cast<size_t>(row)];
        }

        const float diff = std::fabs(got[out_i] - acc);
        mean_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = out_i;
        }
        if (diff > args.tolerance) {
            ++bad;
        }
    }

    mean_abs /= static_cast<double>(samples);
    std::cout << std::format(
        "check sampled={} total={} max_abs={:.9g} mean_abs={:.9g} bad={} max_i={}\n",
        samples,
        total,
        max_abs,
        mean_abs,
        bad,
        max_i);
    return bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    if (args.show_help) {
        usage(argv[0]);
        return 0;
    }
    if (args.warmup < 0 || args.iters <= 0) {
        std::cerr << "--warmup must be non-negative and --iters must be positive\n";
        return 2;
    }

    const size_t weight_elems = static_cast<size_t>(args.k * args.rows);
    const size_t input_elems = static_cast<size_t>(args.k * args.cols * args.batches);
    const size_t output_elems = static_cast<size_t>(args.rows * args.cols * args.batches);
    auto weights_f32 = make_input(weight_elems, 11);
    auto input_f32 = make_input(input_elems, 12);
    auto bias_f32 = make_input(static_cast<size_t>(args.rows), 13);

#ifdef GGML_USE_CUDA
    const int custom_modes = (args.custom_cuda ? 1 : 0) + (args.cublaslt_bias ? 1 : 0) +
                             (args.bf16_cublaslt ? 1 : 0) + (args.gelu_quant_ds4 ? 1 : 0);
    if (custom_modes > 1) {
        std::cerr << "--custom-cuda, --cublaslt-bias, --bf16-cublaslt, and --gelu-quant-ds4 are "
                     "mutually exclusive\n";
        return 2;
    }

    if (args.gelu_quant_ds4) {
        Sam3GeluQuantDs4BenchResult custom{};
        const int64_t quant_cols = args.cols * args.batches;
        auto quant_bias_f32 =
            args.bias ? make_input(static_cast<size_t>(args.k), 14) : std::vector<float>{};
        const int status = sam3_gelu_quant_ds4_bench(input_f32.data(),
                                                     args.bias ? quant_bias_f32.data() : nullptr,
                                                     args.k,
                                                     quant_cols,
                                                     args.warmup,
                                                     args.iters,
                                                     &custom);
        if (status != 0) {
            std::cerr << std::format("GELU+q8_1 DS4 bench failed: {}\n", status);
            return status;
        }
        std::cout << std::format(
            "backend=CUDA-GELU-QUANT-DS4 rows={} cols={} bias={} warmup={} iters={} "
            "separate_mean_ms={:.6f} separate_median_ms={:.6f} separate_p95_ms={:.6f} "
            "fused_mean_ms={:.6f} fused_median_ms={:.6f} fused_p95_ms={:.6f} "
            "speedup_by_median_percent={:.3f} max_abs_output_delta={:.9g} q8_diff_bytes={}\n",
            args.k,
            quant_cols,
            args.bias ? 1 : 0,
            args.warmup,
            args.iters,
            custom.separate.mean_ms,
            custom.separate.median_ms,
            custom.separate.p95_ms,
            custom.fused.mean_ms,
            custom.fused.median_ms,
            custom.fused.p95_ms,
            100.0 * (custom.separate.median_ms - custom.fused.median_ms) /
                custom.separate.median_ms,
            custom.max_abs_output_delta,
            custom.q8_diff_bytes);
        return custom.max_abs_output_delta == 0.0f && custom.q8_diff_bytes == 0 ? 0 : 1;
    }

    if (args.custom_cuda) {
        std::vector<float> got(output_elems);
        Sam3F32BatchedCustomResult custom{};
        const int status = sam3_f32_batched_custom_bench(weights_f32.data(),
                                                         input_f32.data(),
                                                         args.bias ? bias_f32.data() : nullptr,
                                                         got.data(),
                                                         args.k,
                                                         args.rows,
                                                         args.cols,
                                                         args.batches,
                                                         args.warmup,
                                                         args.iters,
                                                         &custom);
        if (status != 0) {
            std::cerr << std::format("custom CUDA bench failed: {}\n", status);
            return status;
        }
        if (args.check && !check_reference(args, weights_f32, input_f32, bias_f32, got)) {
            return 1;
        }
        std::cout << std::format(
            "backend=CUDA-CUSTOM k={} rows={} cols={} batches={} bias={} warmup={} iters={} "
            "mean_ms={:.6f} median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} max_ms={:.6f}\n",
            args.k,
            args.rows,
            args.cols,
            args.batches,
            args.bias ? 1 : 0,
            args.warmup,
            args.iters,
            custom.mean_ms,
            custom.median_ms,
            custom.p95_ms,
            custom.min_ms,
            custom.max_ms);
        return 0;
    }

    if (args.cublaslt_bias) {
        std::vector<float> got(output_elems);
        Sam3F32BatchedCustomResult custom{};
        const int status = sam3_f32_batched_cublaslt_bench(weights_f32.data(),
                                                           input_f32.data(),
                                                           args.bias ? bias_f32.data() : nullptr,
                                                           got.data(),
                                                           args.k,
                                                           args.rows,
                                                           args.cols,
                                                           args.batches,
                                                           args.warmup,
                                                           args.iters,
                                                           args.cublaslt_fast_tf32,
                                                           &custom);
        if (status != 0) {
            std::cerr << std::format("cuBLASLt bias bench failed: {}\n", status);
            return status;
        }
        if (args.check && !check_reference(args, weights_f32, input_f32, bias_f32, got)) {
            return 1;
        }
        std::cout << std::format(
            "backend={} k={} rows={} cols={} batches={} bias={} warmup={} iters={} "
            "mean_ms={:.6f} median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} max_ms={:.6f}\n",
            args.cublaslt_fast_tf32 ? "CUDA-CUBLASLT-TF32" : "CUDA-CUBLASLT",
            args.k,
            args.rows,
            args.cols,
            args.batches,
            args.bias ? 1 : 0,
            args.warmup,
            args.iters,
            custom.mean_ms,
            custom.median_ms,
            custom.p95_ms,
            custom.min_ms,
            custom.max_ms);
        return 0;
    }

    if (args.bf16_cublaslt) {
        std::vector<float> got(output_elems);
        Sam3F32BatchedCustomResult custom{};
        const Sam3Bf16CublasLtOptions options{
            .direct_bf16_dst = args.bf16_direct_dst,
            .gelu_epilogue = args.gelu_epilogue,
            .compute = args.bf16_fast_compute ? Sam3Bf16CublasLtCompute::fast_16bf
                                              : Sam3Bf16CublasLtCompute::fp32,
        };
        const int status = sam3_bf16_batched_cublaslt_bench(weights_f32.data(),
                                                            input_f32.data(),
                                                            args.bias ? bias_f32.data() : nullptr,
                                                            got.data(),
                                                            args.k,
                                                            args.rows,
                                                            args.cols,
                                                            args.batches,
                                                            args.warmup,
                                                            args.iters,
                                                            options,
                                                            &custom);
        if (status != 0) {
            std::cerr << std::format("BF16 cuBLASLt bench failed: {}\n", status);
            return status;
        }
        std::cout << std::format(
            "backend=CUDA-CUBLASLT-BF16 k={} rows={} cols={} batches={} bias={} "
            "direct_bf16_dst={} fast_16bf={} gelu_epilogue={} warmup={} iters={} "
            "mean_ms={:.6f} median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} max_ms={:.6f}\n",
            args.k,
            args.rows,
            args.cols,
            args.batches,
            args.bias ? 1 : 0,
            args.bf16_direct_dst ? 1 : 0,
            args.bf16_fast_compute ? 1 : 0,
            args.gelu_epilogue ? 1 : 0,
            args.warmup,
            args.iters,
            custom.mean_ms,
            custom.median_ms,
            custom.p95_ms,
            custom.min_ms,
            custom.max_ms);
        return 0;
    }
#else
    if (args.custom_cuda || args.cublaslt_bias || args.bf16_cublaslt || args.gelu_quant_ds4) {
        std::cerr << "custom CUDA backend is not compiled in\n";
        return 1;
    }
#endif

    auto backend = create_backend(args);
    if (backend == nullptr) {
        return 1;
    }
    auto ctx = make_context();
    if (ctx == nullptr) {
        std::cerr << "failed to create ggml context\n";
        return 1;
    }

    ggml_tensor* w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, args.k, args.rows);
    ggml_tensor* x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, args.k, args.cols, args.batches);
    ggml_tensor* bias =
        args.bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, args.rows) : nullptr;
    ggml_set_input(w);
    ggml_set_input(x);
    if (bias != nullptr) {
        ggml_set_input(bias);
    }

    ggml_tensor* out = ggml_mul_mat(ctx.get(), w, x);
    if (bias != nullptr) {
        out = ggml_add(ctx.get(), out, bias);
    }
    ggml_set_output(out);

    ggml_cgraph* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    auto alloc = GallocrPtr{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()))};
    if (alloc == nullptr || !ggml_gallocr_reserve(alloc.get(), graph) ||
        !ggml_gallocr_alloc_graph(alloc.get(), graph)) {
        std::cerr << "failed to allocate graph\n";
        return 1;
    }

    ggml_backend_tensor_set(w, weights_f32.data(), 0, weights_f32.size() * sizeof(float));
    ggml_backend_tensor_set(x, input_f32.data(), 0, input_f32.size() * sizeof(float));
    if (bias != nullptr) {
        ggml_backend_tensor_set(bias, bias_f32.data(), 0, bias_f32.size() * sizeof(float));
    }

    for (int i = 0; i < args.warmup; ++i) {
        if (!std::isfinite(run_once_ms(backend.get(), graph))) {
            return 1;
        }
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.iters));
    for (int i = 0; i < args.iters; ++i) {
        const double ms = run_once_ms(backend.get(), graph);
        if (!std::isfinite(ms)) {
            return 1;
        }
        samples.push_back(ms);
    }

    if (args.check) {
        std::vector<float> got(output_elems);
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
        if (!check_reference(args, weights_f32, input_f32, bias_f32, got)) {
            return 1;
        }
    }

    const auto result = summarize(std::move(samples));
    std::cout << std::format(
        "backend={} k={} rows={} cols={} batches={} bias={} warmup={} iters={} mean_ms={:.6f} "
        "median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} max_ms={:.6f}\n",
        args.cuda ? "CUDA" : "CPU",
        args.k,
        args.rows,
        args.cols,
        args.batches,
        args.bias ? 1 : 0,
        args.warmup,
        args.iters,
        result.mean_ms,
        result.median_ms,
        result.p95_ms,
        result.min_ms,
        result.max_ms);
    return 0;
}
