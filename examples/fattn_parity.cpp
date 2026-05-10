#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace {

struct Args {
    int d = 56;
    int n = 196;
    int heads = 8;
    int batch = 1;
    int sample_queries = 0;
    float tolerance = 4.0e-2f;
    bool cuda = true;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--cpu|--cuda] [--d <head_dim>] [--n <tokens>] "
                 "[--heads <n>] [--batch <n>] [--sample-queries <n>] [--tolerance <f>]\n",
                 argv0);
}

static bool parse_int(const char* s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || v <= 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_float(const char* s, float& out) {
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (!end || *end != '\0' || !std::isfinite(v) || v <= 0.0f) {
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
                std::fprintf(stderr, "missing value for %s\n", name);
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
        } else if (arg == "--tolerance") {
            const char* v = need_value("--tolerance");
            if (!v || !parse_float(v, args.tolerance)) {
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

static size_t index4(int d, int n, int h, int b, int D, int N, int H) {
    return static_cast<size_t>(((b * H + h) * N + n) * D + d);
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
        const int idx = args.sample_queries == 1 ? 0 : (int64_t)i * (args.n - 1) / (args.sample_queries - 1);
        if (out.empty() || out.back() != idx) {
            out.push_back(idx);
        }
    }
    return out;
}

static ggml_backend_t create_backend(const Args& args) {
#ifdef GGML_USE_CUDA
    if (args.cuda) {
        return ggml_backend_cuda_init(0);
    }
#else
    if (args.cuda) {
        std::fprintf(stderr, "CUDA backend is not compiled in\n");
        return nullptr;
    }
#endif
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend) {
        ggml_backend_cpu_set_n_threads(backend, 4);
    }
    return backend;
}

static bool run_ggml_attention(const Args& args,
                               const std::vector<float>& q_data,
                               const std::vector<float>& k_data,
                               const std::vector<float>& v_data,
                               std::vector<float>& out_data) {
    ggml_backend_t backend = create_backend(args);
    if (!backend) {
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, args.d, args.n, args.heads, args.batch);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, args.d, args.n, args.heads, args.batch);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, args.d, args.n, args.heads, args.batch);
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);

    const float scale = 1.0f / std::sqrt(static_cast<float>(args.d));
    ggml_tensor* out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_set_output(out);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "failed to allocate graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    out_data.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    const size_t elements = static_cast<size_t>(args.d) * args.n * args.heads * args.batch;
    auto q = make_input(elements, 1);
    auto k = make_input(elements, 2);
    auto v = make_input(elements, 3);

    // CUDA FlashAttention converts K/V to fp16 for this path. Match that in the
    // scalar reference so this test catches kernel bugs instead of conversion noise.
    auto k_ref = args.cuda ? fp16_rounded(k) : k;
    auto v_ref = args.cuda ? fp16_rounded(v) : v;
    const auto query_indices = make_query_indices(args);
    auto ref = reference_attention(q, k_ref, v_ref, args, query_indices);

    std::vector<float> got;
    if (!run_ggml_attention(args, q, k, v, got)) {
        return 1;
    }
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "size mismatch: got=%zu ref=%zu\n", got.size(), ref.size());
        return 1;
    }

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    size_t max_i = 0;
    size_t bad = 0;
    size_t checked = 0;
    for (int b = 0; b < args.batch; ++b) {
        for (int h = 0; h < args.heads; ++h) {
            for (const int nq : query_indices) {
                for (int d = 0; d < args.d; ++d) {
                    const size_t i = index4(d, nq, h, b, args.d, args.n, args.heads);
                    const float diff = std::fabs(got[i] - ref[i]);
                    mean_abs += diff;
                    if (diff > max_abs) {
                        max_abs = diff;
                        max_i = i;
                    }
                    if (diff > args.tolerance) {
                        ++bad;
                    }
                    ++checked;
                }
            }
        }
    }
    mean_abs /= static_cast<double>(checked);

    std::printf("backend=%s D=%d N=%d heads=%d batch=%d sampled_q=%zu max_abs=%.9g mean_abs=%.9g bad=%zu/%zu max_i=%zu\n",
                args.cuda ? "CUDA" : "CPU",
                args.d,
                args.n,
                args.heads,
                args.batch,
                query_indices.size(),
                max_abs,
                mean_abs,
                bad,
                checked,
                max_i);

    if (bad != 0) {
        std::fprintf(stderr,
                     "parity failed: max_abs %.9g exceeds tolerance %.9g at index %zu "
                     "(got %.9g ref %.9g)\n",
                     max_abs,
                     args.tolerance,
                     max_i,
                     got[max_i],
                     ref[max_i]);
        return 1;
    }
    return 0;
}
