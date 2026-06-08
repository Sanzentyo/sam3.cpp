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
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

struct Args {
    int d = 56;
    int run_d = 0;
    int n = 196;
    int heads = 8;
    int batch = 1;
    int sample_queries = 0;
    int warmup = 0;
    int iters = 0;
    float tolerance = 4.0e-2f;
    bool cuda = true;
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
        "Usage: {} [--cpu|--cuda] [--d <head_dim>] [--n <tokens>] "
        "[--run-d <head_dim>] [--heads <n>] [--batch <n>] [--sample-queries <n>] "
        "[--warmup <n>] [--iters <n>] [--tolerance <f>]\n",
        argv0);
}

static bool parse_int_impl(const char* s, int& out, bool allow_zero) {
    const std::string_view sv{s};
    int v = 0;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || ptr != sv.data() + sv.size() || v < 0 || (!allow_zero && v == 0)) {
        return false;
    }
    out = v;
    return true;
}

static bool parse_int(const char* s, int& out) {
    return parse_int_impl(s, out, false);
}

static bool parse_nonnegative_int(const char* s, int& out) {
    return parse_int_impl(s, out, true);
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
        } else if (arg == "--d") {
            const char* v = need_value("--d");
            if (!v || !parse_int(v, args.d)) {
                return false;
            }
        } else if (arg == "--n") {
            const char* v = need_value("--n");
            if (!v || !parse_int(v, args.n)) {
                return false;
            }
        } else if (arg == "--run-d") {
            const char* v = need_value("--run-d");
            if (!v || !parse_int(v, args.run_d)) {
                return false;
            }
        } else if (arg == "--heads") {
            const char* v = need_value("--heads");
            if (!v || !parse_int(v, args.heads)) {
                return false;
            }
        } else if (arg == "--batch") {
            const char* v = need_value("--batch");
            if (!v || !parse_int(v, args.batch)) {
                return false;
            }
        } else if (arg == "--sample-queries") {
            const char* v = need_value("--sample-queries");
            if (!v || !parse_int(v, args.sample_queries)) {
                return false;
            }
        } else if (arg == "--warmup") {
            const char* v = need_value("--warmup");
            if (!v || !parse_nonnegative_int(v, args.warmup)) {
                return false;
            }
        } else if (arg == "--iters") {
            const char* v = need_value("--iters");
            if (!v || !parse_int(v, args.iters)) {
                return false;
            }
        } else if (arg == "--tolerance") {
            const char* v = need_value("--tolerance");
            if (!v || !parse_float(v, args.tolerance)) {
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

static size_t index4(int d, int n, int h, int b, int D, int N, int H) {
    return static_cast<size_t>(((b * H + h) * N + n) * D + d);
}

struct Index4 {
    int d = 0;
    int n = 0;
    int h = 0;
    int b = 0;
};

static Index4 decode_index4(size_t i, int D, int N, int H) {
    Index4 out;
    out.d = static_cast<int>(i % static_cast<size_t>(D));
    i /= static_cast<size_t>(D);
    out.n = static_cast<int>(i % static_cast<size_t>(N));
    i /= static_cast<size_t>(N);
    out.h = static_cast<int>(i % static_cast<size_t>(H));
    i /= static_cast<size_t>(H);
    out.b = static_cast<int>(i);
    return out;
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

static std::vector<float> fp16_rounded(const std::vector<float>& src) {
    std::vector<ggml_fp16_t> tmp(src.size());
    std::vector<float> dst(src.size());
    ggml_fp32_to_fp16_row(src.data(), tmp.data(), static_cast<int64_t>(src.size()));
    ggml_fp16_to_fp32_row(tmp.data(), dst.data(), static_cast<int64_t>(dst.size()));
    return dst;
}

static std::vector<float> pad_head_dim(const std::vector<float>& src, const Args& args, int run_d) {
    if (run_d == args.d) {
        return src;
    }
    std::vector<float> out(static_cast<size_t>(run_d) * args.n * args.heads * args.batch, 0.0f);
    for (int b = 0; b < args.batch; ++b) {
        for (int h = 0; h < args.heads; ++h) {
            for (int n = 0; n < args.n; ++n) {
                for (int d = 0; d < args.d; ++d) {
                    out[index4(d, n, h, b, run_d, args.n, args.heads)] =
                        src[index4(d, n, h, b, args.d, args.n, args.heads)];
                }
            }
        }
    }
    return out;
}

static std::vector<float> reference_attention(const std::vector<float>& q,
                                              const std::vector<float>& k,
                                              const std::vector<float>& v,
                                              const Args& args,
                                              const std::vector<int>& query_indices) {
    const int D = args.d;
    const int N = args.n;
    const int H = args.heads;
    const int B = args.batch;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> out(static_cast<size_t>(D) * N * H * B);
    std::vector<float> scores(N);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (const int nq : query_indices) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (int nk = 0; nk < N; ++nk) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += q[index4(d, nq, h, b, D, N, H)] * k[index4(d, nk, h, b, D, N, H)];
                    }
                    scores[nk] = dot * scale;
                    max_score = std::max(max_score, scores[nk]);
                }

                float denom = 0.0f;
                for (int nk = 0; nk < N; ++nk) {
                    scores[nk] = std::exp(scores[nk] - max_score);
                    denom += scores[nk];
                }

                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int nk = 0; nk < N; ++nk) {
                        acc += scores[nk] * v[index4(d, nk, h, b, D, N, H)];
                    }
                    out[index4(d, nq, h, b, D, N, H)] = acc / denom;
                }
            }
        }
    }
    return out;
}

static std::vector<int> make_query_indices(const Args& args) {
    if (args.sample_queries <= 0 || args.sample_queries >= args.n) {
        std::vector<int> all(args.n);
        for (int i = 0; i < args.n; ++i) {
            all[i] = i;
        }
        return all;
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(args.sample_queries));
    for (int i = 0; i < args.sample_queries; ++i) {
        const int idx =
            args.sample_queries == 1 ? 0 : (int64_t) i * (args.n - 1) / (args.sample_queries - 1);
        if (out.empty() || out.back() != idx) {
            out.push_back(idx);
        }
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
        ggml_backend_cpu_set_n_threads(backend.get(), 4);
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

static bool run_ggml_attention(const Args& args,
                               int run_d,
                               const std::vector<float>& q_data,
                               const std::vector<float>& k_data,
                               const std::vector<float>& v_data,
                               std::vector<float>& out_data,
                               BenchResult* timing) {
    auto backend = create_backend(args);
    if (backend == nullptr) {
        return false;
    }

    auto ctx = make_context();
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor* q =
        ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, run_d, args.n, args.heads, args.batch);
    ggml_tensor* k =
        ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, run_d, args.n, args.heads, args.batch);
    ggml_tensor* v =
        ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, run_d, args.n, args.heads, args.batch);
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);

    const float scale = 1.0f / std::sqrt(static_cast<float>(args.d));
    ggml_tensor* out = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_set_output(out);

    ggml_cgraph* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    auto alloc = GallocrPtr{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get()))};
    if (alloc == nullptr || !ggml_gallocr_reserve(alloc.get(), graph) ||
        !ggml_gallocr_alloc_graph(alloc.get(), graph)) {
        std::cerr << "failed to allocate graph\n";
        return false;
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));

    if (args.iters > 0) {
        for (int i = 0; i < args.warmup; ++i) {
            if (!std::isfinite(run_once_ms(backend.get(), graph))) {
                return false;
            }
        }

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(args.iters));
        for (int i = 0; i < args.iters; ++i) {
            const double ms = run_once_ms(backend.get(), graph);
            if (!std::isfinite(ms)) {
                return false;
            }
            samples.push_back(ms);
        }
        if (timing != nullptr) {
            *timing = summarize(std::move(samples));
        }
    } else {
        if (!std::isfinite(run_once_ms(backend.get(), graph))) {
            return false;
        }
    }

    out_data.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));
    return true;
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
    const int run_d = args.run_d > 0 ? args.run_d : args.d;
    if (run_d < args.d) {
        std::cerr << std::format("--run-d must be >= --d, got {} < {}\n", run_d, args.d);
        return 2;
    }

    const size_t elements = static_cast<size_t>(args.d) * args.n * args.heads * args.batch;
    auto q = make_input(elements, 1);
    auto k = make_input(elements, 2);
    auto v = make_input(elements, 3);
    auto q_run = pad_head_dim(q, args, run_d);
    auto k_run = pad_head_dim(k, args, run_d);
    auto v_run = pad_head_dim(v, args, run_d);

    // CUDA FlashAttention converts K/V to fp16 for this path. Match that in the
    // scalar reference so this test catches kernel bugs instead of conversion noise.
    auto k_ref = args.cuda ? fp16_rounded(k) : k;
    auto v_ref = args.cuda ? fp16_rounded(v) : v;
    const auto query_indices = make_query_indices(args);
    auto ref = reference_attention(q, k_ref, v_ref, args, query_indices);

    std::vector<float> got;
    BenchResult timing;
    if (!run_ggml_attention(args, run_d, q_run, k_run, v_run, got, &timing)) {
        return 1;
    }
    const size_t expected_got_size = static_cast<size_t>(run_d) * args.n * args.heads * args.batch;
    if (got.size() != expected_got_size) {
        std::cerr << std::format(
            "size mismatch: got={} expected={}\n", got.size(), expected_got_size);
        return 1;
    }

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    size_t max_i = 0;
    size_t first_nonfinite_i = 0;
    float max_got = 0.0f;
    float max_ref = 0.0f;
    float first_nonfinite_got = 0.0f;
    float first_nonfinite_ref = 0.0f;
    size_t bad = 0;
    size_t nonfinite = 0;
    size_t checked = 0;
    for (int b = 0; b < args.batch; ++b) {
        for (int h = 0; h < args.heads; ++h) {
            for (const int nq : query_indices) {
                for (int d = 0; d < args.d; ++d) {
                    const size_t got_i = index4(d, nq, h, b, run_d, args.n, args.heads);
                    const size_t ref_i = index4(d, nq, h, b, args.d, args.n, args.heads);
                    if (!std::isfinite(got[got_i]) || !std::isfinite(ref[ref_i])) {
                        ++nonfinite;
                        if (nonfinite == 1) {
                            first_nonfinite_i = got_i;
                            first_nonfinite_got = got[got_i];
                            first_nonfinite_ref = ref[ref_i];
                        }
                        ++checked;
                        continue;
                    }
                    const float diff = std::fabs(got[got_i] - ref[ref_i]);
                    mean_abs += diff;
                    if (diff > max_abs) {
                        max_abs = diff;
                        max_i = got_i;
                        max_got = got[got_i];
                        max_ref = ref[ref_i];
                    }
                    if (diff > args.tolerance) {
                        ++bad;
                    }
                    ++checked;
                }
            }
        }
    }
    mean_abs = nonfinite == checked ? std::numeric_limits<double>::quiet_NaN()
                                    : mean_abs / static_cast<double>(checked - nonfinite);

    std::cout << std::format(
        "backend={} D={} run_D={} N={} heads={} batch={} sampled_q={} max_abs={:.9g} "
        "mean_abs={:.9g} bad={}/{} nonfinite={} max_i={}",
        args.cuda ? "CUDA" : "CPU",
        args.d,
        run_d,
        args.n,
        args.heads,
        args.batch,
        query_indices.size(),
        max_abs,
        mean_abs,
        bad,
        checked,
        nonfinite,
        max_i);
    if (args.iters > 0) {
        std::cout << std::format(
            " warmup={} iters={} mean_ms={:.6f} median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} "
            "max_ms={:.6f}",
            args.warmup,
            args.iters,
            timing.mean_ms,
            timing.median_ms,
            timing.p95_ms,
            timing.min_ms,
            timing.max_ms);
    }
    std::cout << "\n";

    if (bad != 0 || nonfinite != 0) {
        const Index4 max_coord = decode_index4(max_i, run_d, args.n, args.heads);
        std::cerr << std::format(
            "parity failed: max_abs {:.9g} exceeds tolerance {:.9g} at index {} "
            "d/n/h/b={}/{}/{}/{} (got {:.9g} ref {:.9g}), nonfinite={}",
            max_abs,
            args.tolerance,
            max_i,
            max_coord.d,
            max_coord.n,
            max_coord.h,
            max_coord.b,
            max_got,
            max_ref,
            nonfinite);
        if (nonfinite != 0) {
            const Index4 first_nonfinite_coord =
                decode_index4(first_nonfinite_i, run_d, args.n, args.heads);
            std::cerr << std::format(
                " first_nonfinite_index={} d/n/h/b={}/{}/{}/{} (got {} ref {})",
                first_nonfinite_i,
                first_nonfinite_coord.d,
                first_nonfinite_coord.n,
                first_nonfinite_coord.h,
                first_nonfinite_coord.b,
                first_nonfinite_got,
                first_nonfinite_ref);
        }
        std::cerr << "\n";
        return 1;
    }
    return 0;
}
