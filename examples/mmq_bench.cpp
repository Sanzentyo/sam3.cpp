#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
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
#include <numbers>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

namespace {

struct Args {
    int64_t k = 448;
    int64_t rows = 1792;
    int64_t fc2_rows = 448;
    int64_t cols = 4096;
    int64_t channels = 1;
    int warmup = 3;
    int iters = 20;
    int threads = 4;
    ggml_type type = GGML_TYPE_Q8_0;
    float tolerance = 2.5e-1f;
    bool cuda = true;
    bool check = false;
    bool bias = false;
    bool gelu = false;
    bool two_stage = false;
    bool two_stage_cont = false;
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
        "Usage: {} [--cpu|--cuda] [--type q8_0|q4_0|q4_1] [--k <n>] [--rows <n>] "
        "[--two-stage] [--fc2-rows <n>] "
        "[--cols <n>] [--channels <n>] [--warmup <n>] [--iters <n>] [--threads <n>] "
        "[--check] [--bias] [--gelu] [--two-stage-cont] [--tolerance <f>]\n",
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

static bool parse_nonnegative_int(const char* s, int& out) {
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

static bool parse_type(std::string_view name, ggml_type& out) {
    if (name == "q8_0") {
        out = GGML_TYPE_Q8_0;
        return true;
    }
    if (name == "q4_0") {
        out = GGML_TYPE_Q4_0;
        return true;
    }
    if (name == "q4_1") {
        out = GGML_TYPE_Q4_1;
        return true;
    }
    return false;
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
        } else if (arg == "--check") {
            args.check = true;
        } else if (arg == "--bias") {
            args.bias = true;
        } else if (arg == "--gelu") {
            args.gelu = true;
        } else if (arg == "--two-stage") {
            args.two_stage = true;
        } else if (arg == "--two-stage-cont") {
            args.two_stage_cont = true;
        } else if (arg == "--type") {
            const char* v = need_value("--type");
            if (v == nullptr || !parse_type(v, args.type)) {
                return false;
            }
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
        } else if (arg == "--fc2-rows") {
            const char* v = need_value("--fc2-rows");
            if (v == nullptr || !parse_int64(v, args.fc2_rows)) {
                return false;
            }
        } else if (arg == "--cols") {
            const char* v = need_value("--cols");
            if (v == nullptr || !parse_int64(v, args.cols)) {
                return false;
            }
        } else if (arg == "--channels") {
            const char* v = need_value("--channels");
            if (v == nullptr || !parse_int64(v, args.channels)) {
                return false;
            }
        } else if (arg == "--warmup") {
            const char* v = need_value("--warmup");
            if (v == nullptr || !parse_nonnegative_int(v, args.warmup)) {
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
    return (u - 0.5f) * 0.25f;
}

static std::vector<float> make_input(size_t n, int salt) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = deterministic_value(i, salt);
    }
    return out;
}

static float gelu(float x) {
    return 0.5f * x *
           (1.0f +
            std::tanh(std::sqrt(2.0f / std::numbers::pi_v<float>) * (x + 0.044715f * x * x * x)));
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
    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return ContextPtr{ggml_init(params)};
}

static std::vector<uint8_t> quantize_weights(ggml_type type,
                                             int64_t k,
                                             int64_t rows,
                                             std::span<const float> f32) {
    const size_t bytes = ggml_row_size(type, k) * static_cast<size_t>(rows);
    std::vector<uint8_t> out(bytes);
    ggml_quantize_chunk(type, f32.data(), out.data(), 0, rows, k, nullptr);
    return out;
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

static bool check_reference(std::span<const float> expected,
                            std::span<const float> got,
                            float tolerance) {
    const size_t total = expected.size();
    if (total > 4ull * 1024ull * 1024ull) {
        std::cerr << std::format(
            "--check is capped at 4194304 output elements, got {}. Reduce --rows/--cols.\n", total);
        return false;
    }

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    size_t max_i = 0;
    size_t bad = 0;
    size_t nonfinite_expected = 0;
    size_t nonfinite_got = 0;
    for (size_t i = 0; i < total; ++i) {
        if (!std::isfinite(expected[i])) {
            ++nonfinite_expected;
        }
        if (!std::isfinite(got[i])) {
            ++nonfinite_got;
        }
        const float diff = std::fabs(got[i] - expected[i]);
        if (!std::isfinite(diff)) {
            ++bad;
            max_i = i;
            max_abs = std::numeric_limits<float>::infinity();
            continue;
        }
        mean_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_i = i;
        }
        if (diff > tolerance) {
            ++bad;
        }
    }

    mean_abs /= static_cast<double>(total);
    std::cout << std::format(
        "check max_abs={:.9g} mean_abs={:.9g} bad={}/{} max_i={} nonfinite_expected={} "
        "nonfinite_got={}\n",
        max_abs,
        mean_abs,
        bad,
        total,
        max_i,
        nonfinite_expected,
        nonfinite_got);
    return bad == 0;
}

static std::vector<float> reference_single(const Args& args,
                                           std::span<const float> weights,
                                           std::span<const float> input,
                                           std::span<const float> bias) {
    std::vector<float> out(static_cast<size_t>(args.rows * args.cols * args.channels));
    for (int64_t channel = 0; channel < args.channels; ++channel) {
        for (int64_t col = 0; col < args.cols; ++col) {
            for (int64_t row = 0; row < args.rows; ++row) {
                float acc = 0.0f;
                for (int64_t kk = 0; kk < args.k; ++kk) {
                    acc += weights[static_cast<size_t>(row * args.k + kk)] *
                           input[static_cast<size_t>((channel * args.cols + col) * args.k + kk)];
                }
                if (args.bias) {
                    acc += bias[static_cast<size_t>(row)];
                }
                if (args.gelu) {
                    acc = gelu(acc);
                }
                out[static_cast<size_t>((channel * args.cols + col) * args.rows + row)] = acc;
            }
        }
    }
    return out;
}

static std::vector<float> reference_two_stage_cpu_graph(const Args& args,
                                                        std::span<const uint8_t> w1_q,
                                                        std::span<const uint8_t> w2_q,
                                                        std::span<const float> input,
                                                        std::span<const float> bias1,
                                                        std::span<const float> bias2) {
    BackendPtr backend{ggml_backend_cpu_init()};
    if (backend == nullptr) {
        return {};
    }
    ggml_backend_cpu_set_n_threads(backend.get(), args.threads);

    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ContextPtr ctx{ggml_init(params)};
    if (ctx == nullptr) {
        return {};
    }

    ggml_tensor* w1 = ggml_new_tensor_2d(ctx.get(), args.type, args.k, args.rows);
    ggml_tensor* x =
        args.channels == 1
            ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, args.k, args.cols)
            : ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, args.k, args.cols, args.channels);
    ggml_tensor* bias1_tensor =
        args.bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, args.rows) : nullptr;
    ggml_tensor* out = ggml_mul_mat(ctx.get(), w1, x);
    if (bias1_tensor != nullptr) {
        out = ggml_add(ctx.get(), out, bias1_tensor);
    }
    out = ggml_gelu(ctx.get(), out);
    if (args.two_stage_cont) {
        out = ggml_cont(ctx.get(), out);
    }

    ggml_tensor* w2 = ggml_new_tensor_2d(ctx.get(), args.type, args.rows, args.fc2_rows);
    ggml_tensor* bias2_tensor =
        args.bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, args.fc2_rows) : nullptr;
    out = ggml_mul_mat(ctx.get(), w2, out);
    if (bias2_tensor != nullptr) {
        out = ggml_add(ctx.get(), out, bias2_tensor);
    }
    ggml_set_output(out);

    ggml_cgraph* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    GallocrPtr alloc{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()))};
    if (alloc == nullptr || !ggml_gallocr_reserve(alloc.get(), graph) ||
        !ggml_gallocr_alloc_graph(alloc.get(), graph)) {
        return {};
    }

    ggml_backend_tensor_set(w1, w1_q.data(), 0, w1_q.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (bias1_tensor != nullptr) {
        ggml_backend_tensor_set(bias1_tensor, bias1.data(), 0, bias1.size() * sizeof(float));
    }
    ggml_backend_tensor_set(w2, w2_q.data(), 0, w2_q.size());
    if (bias2_tensor != nullptr) {
        ggml_backend_tensor_set(bias2_tensor, bias2.data(), 0, bias2.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        return {};
    }

    std::vector<float> expected(static_cast<size_t>(args.fc2_rows * args.cols * args.channels));
    ggml_backend_tensor_get(out, expected.data(), 0, expected.size() * sizeof(float));
    return expected;
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
    if (args.k % static_cast<int64_t>(ggml_blck_size(args.type)) != 0) {
        std::cerr << std::format("--k={} must be divisible by block size {} for {}\n",
                                 args.k,
                                 ggml_blck_size(args.type),
                                 ggml_type_name(args.type));
        return 2;
    }
    if (args.two_stage && args.rows % static_cast<int64_t>(ggml_blck_size(args.type)) != 0) {
        std::cerr << std::format(
            "--rows={} must be divisible by block size {} for a quantized fc2 input\n",
            args.rows,
            ggml_blck_size(args.type));
        return 2;
    }
    const size_t weight_elems = static_cast<size_t>(args.k * args.rows);
    const size_t fc2_weight_elems = static_cast<size_t>(args.rows * args.fc2_rows);
    const size_t input_elems = static_cast<size_t>(args.k * args.cols * args.channels);
    const size_t output_elems = static_cast<size_t>((args.two_stage ? args.fc2_rows : args.rows) *
                                                    args.cols * args.channels);
    auto weights_f32 = make_input(weight_elems, 1);
    auto fc2_weights_f32 = make_input(fc2_weight_elems, 4);
    auto input_f32 = make_input(input_elems, 2);
    auto bias_f32 = make_input(static_cast<size_t>(args.rows), 3);
    auto fc2_bias_f32 = make_input(static_cast<size_t>(args.fc2_rows), 5);
    auto weights_q = quantize_weights(args.type, args.k, args.rows, weights_f32);
    auto fc2_weights_q = quantize_weights(args.type, args.rows, args.fc2_rows, fc2_weights_f32);

    auto backend = create_backend(args);
    if (backend == nullptr) {
        return 1;
    }
    auto ctx = make_context();
    if (ctx == nullptr) {
        std::cerr << "failed to create ggml context\n";
        return 1;
    }

    ggml_tensor* w1 = ggml_new_tensor_2d(ctx.get(), args.type, args.k, args.rows);
    ggml_tensor* x =
        args.channels == 1
            ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, args.k, args.cols)
            : ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, args.k, args.cols, args.channels);
    ggml_tensor* bias1 =
        args.bias ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, args.rows) : nullptr;
    ggml_set_input(w1);
    ggml_set_input(x);
    if (bias1 != nullptr) {
        ggml_set_input(bias1);
    }
    ggml_tensor* out = ggml_mul_mat(ctx.get(), w1, x);
    if (bias1 != nullptr) {
        out = ggml_add(ctx.get(), out, bias1);
    }
    if (args.gelu || args.two_stage) {
        out = ggml_gelu(ctx.get(), out);
    }
    ggml_tensor* w2 = nullptr;
    ggml_tensor* bias2 = nullptr;
    if (args.two_stage) {
        if (args.two_stage_cont) {
            out = ggml_cont(ctx.get(), out);
        }
        w2 = ggml_new_tensor_2d(ctx.get(), args.type, args.rows, args.fc2_rows);
        ggml_set_input(w2);
        if (args.bias) {
            bias2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, args.fc2_rows);
            ggml_set_input(bias2);
        }
        out = ggml_mul_mat(ctx.get(), w2, out);
        if (bias2 != nullptr) {
            out = ggml_add(ctx.get(), out, bias2);
        }
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

    const auto set_inputs = [&] {
        ggml_backend_tensor_set(w1, weights_q.data(), 0, weights_q.size());
        ggml_backend_tensor_set(x, input_f32.data(), 0, input_f32.size() * sizeof(float));
        if (bias1 != nullptr) {
            ggml_backend_tensor_set(bias1, bias_f32.data(), 0, bias_f32.size() * sizeof(float));
        }
        if (w2 != nullptr) {
            ggml_backend_tensor_set(w2, fc2_weights_q.data(), 0, fc2_weights_q.size());
        }
        if (bias2 != nullptr) {
            ggml_backend_tensor_set(
                bias2, fc2_bias_f32.data(), 0, fc2_bias_f32.size() * sizeof(float));
        }
    };

    set_inputs();

    for (int i = 0; i < args.warmup; ++i) {
        if (args.two_stage) {
            set_inputs();
        }
        if (!std::isfinite(run_once_ms(backend.get(), graph))) {
            return 1;
        }
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.iters));
    for (int i = 0; i < args.iters; ++i) {
        if (args.two_stage) {
            set_inputs();
        }
        const double ms = run_once_ms(backend.get(), graph);
        if (!std::isfinite(ms)) {
            return 1;
        }
        samples.push_back(ms);
    }

    std::vector<float> got;
    if (args.check) {
        got.resize(output_elems);
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
        const auto expected =
            args.two_stage ? reference_two_stage_cpu_graph(
                                 args, weights_q, fc2_weights_q, input_f32, bias_f32, fc2_bias_f32)
                           : reference_single(args, weights_f32, input_f32, bias_f32);
        if (expected.empty()) {
            std::cerr << "failed to compute two-stage CPU reference\n";
            return 1;
        }
        if (!check_reference(expected, got, args.tolerance)) {
            return 1;
        }
    }

    const auto result = summarize(std::move(samples));
    std::cout << std::format(
        "backend={} type={} k={} rows={} fc2_rows={} cols={} channels={} bias={} gelu={} "
        "two_stage={} two_stage_cont={} warmup={} iters={} "
        "mean_ms={:.6f} median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} max_ms={:.6f}\n",
        args.cuda ? "CUDA" : "CPU",
        ggml_type_name(args.type),
        args.k,
        args.rows,
        args.fc2_rows,
        args.cols,
        args.channels,
        args.bias ? 1 : 0,
        args.gelu ? 1 : 0,
        args.two_stage ? 1 : 0,
        args.two_stage_cont ? 1 : 0,
        args.warmup,
        args.iters,
        result.mean_ms,
        result.median_ms,
        result.p95_ms,
        result.min_ms,
        result.max_ms);
    return 0;
}
