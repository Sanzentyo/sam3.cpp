#include <cuda_runtime_api.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t FC1_W_UID = 1;
constexpr int64_t INPUT_UID = 2;
constexpr int64_t FC1_BIAS_UID = 3;
constexpr int64_t FC1_OUT_UID = 4;
constexpr int64_t FC2_W_UID = 5;
constexpr int64_t FC2_BIAS_UID = 6;
constexpr int64_t FINAL_OUT_UID = 7;

void cuda_check(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::format("{} failed: {}", expr, cudaGetErrorString(status)));
    }
}

void cudnn_check(cudnnStatus_t status, const char* expr) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::format("{} failed: {}", expr, cudnnGetErrorString(status)));
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)
#define CUDNN_CHECK(expr) cudnn_check((expr), #expr)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    T* get() const { return ptr_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

struct Args {
    int64_t input_dim = 1024;
    int64_t hidden_dim = 4736;
    int64_t output_dim = 1024;
    int64_t cols = 5184;
    int warmup = 5;
    int iters = 30;
    bool full_chain = false;
    bool approx_gelu = false;
    bool row_major = false;
    bool ggml_weight_layout = false;
    bool check = false;
    float tolerance = 5e-3f;
    std::string io_dtype = "bf16";
    std::string fc1_out_dtype = "bf16";
    std::string final_out_dtype = "fp32";
    std::string dump_output;
};

fe::DataType_t parse_dtype(std::string_view value) {
    if (value == "bf16") {
        return fe::DataType_t::BFLOAT16;
    }
    if (value == "fp16") {
        return fe::DataType_t::HALF;
    }
    if (value == "fp32") {
        return fe::DataType_t::FLOAT;
    }
    throw std::runtime_error(std::format("unsupported dtype: {}", value));
}

size_t dtype_size(std::string_view value) {
    if (value == "bf16" || value == "fp16") {
        return 2;
    }
    if (value == "fp32") {
        return 4;
    }
    throw std::runtime_error(std::format("unsupported dtype: {}", value));
}

std::vector<int64_t> matrix_stride(bool row_major, int64_t rows, int64_t cols) {
    if (row_major) {
        return {rows * cols, cols, 1};
    }
    return {rows * cols, 1, rows};
}

std::vector<int64_t> weight_stride(bool row_major,
                                   bool ggml_weight_layout,
                                   int64_t rows,
                                   int64_t cols) {
    if (ggml_weight_layout) {
        return {rows * cols, cols, 1};
    }
    return matrix_stride(row_major, rows, cols);
}

std::vector<int64_t> channel_bias_stride(bool row_major, int64_t rows) {
    if (row_major) {
        return {rows, 1, 1};
    }
    return {rows, 1, rows};
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        auto need_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::format("{} needs a value", name));
            }
            return argv[++i];
        };
        if (key == "--input-dim") {
            args.input_dim = std::stoll(need_value(key));
        } else if (key == "--hidden-dim") {
            args.hidden_dim = std::stoll(need_value(key));
        } else if (key == "--output-dim") {
            args.output_dim = std::stoll(need_value(key));
        } else if (key == "--cols") {
            args.cols = std::stoll(need_value(key));
        } else if (key == "--warmup") {
            args.warmup = std::stoi(need_value(key));
        } else if (key == "--iters") {
            args.iters = std::stoi(need_value(key));
        } else if (key == "--full-chain") {
            args.full_chain = true;
        } else if (key == "--approx-gelu") {
            args.approx_gelu = true;
        } else if (key == "--row-major") {
            args.row_major = true;
        } else if (key == "--ggml-weight-layout") {
            args.ggml_weight_layout = true;
        } else if (key == "--check") {
            args.check = true;
        } else if (key == "--tolerance") {
            args.tolerance = std::stof(need_value(key));
        } else if (key == "--io-dtype") {
            args.io_dtype = need_value(key);
            parse_dtype(args.io_dtype);
        } else if (key == "--fc1-out-dtype") {
            args.fc1_out_dtype = need_value(key);
            parse_dtype(args.fc1_out_dtype);
        } else if (key == "--final-out-dtype") {
            args.final_out_dtype = need_value(key);
            parse_dtype(args.final_out_dtype);
        } else if (key == "--dump-output") {
            args.dump_output = need_value(key);
        } else {
            throw std::runtime_error(std::format("unknown argument: {}", key));
        }
    }
    if (args.input_dim <= 0 || args.hidden_dim <= 0 || args.output_dim <= 0 || args.cols <= 0 ||
        args.warmup < 0 || args.iters <= 0) {
        throw std::runtime_error("all dimensions and iters must be positive");
    }
    return args;
}

uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float pattern_value(size_t index, int modulus, float scale) {
    const int centered =
        static_cast<int>((index * 13 + 7) % static_cast<size_t>(modulus)) - modulus / 2;
    return static_cast<float>(centered) * scale;
}

float gelu(float x, bool approx) {
    if (approx) {
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

std::vector<uint16_t> make_bf16_pattern(size_t count, int modulus, float scale) {
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = f32_to_bf16_bits(pattern_value(i, modulus, scale));
    }
    return data;
}

std::vector<float> make_f32_pattern(size_t count, int modulus, float scale) {
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = pattern_value(i, modulus, scale);
    }
    return data;
}

std::shared_ptr<fe::graph::Graph> create_mlp_graph(const Args& args) {
    constexpr int64_t batch = 1;
    const auto io_dtype = parse_dtype(args.io_dtype);
    const auto fc1_out_dtype = parse_dtype(args.fc1_out_dtype);
    const auto final_out_dtype = parse_dtype(args.final_out_dtype);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto fc1_w = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("fc1_w")
            .set_uid(FC1_W_UID)
            .set_dim({batch, args.hidden_dim, args.input_dim})
            .set_stride(weight_stride(
                args.row_major, args.ggml_weight_layout, args.hidden_dim, args.input_dim))
            .set_data_type(io_dtype));
    auto input =
        graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_uid(INPUT_UID)
                          .set_dim({batch, args.input_dim, args.cols})
                          .set_stride(matrix_stride(args.row_major, args.input_dim, args.cols))
                          .set_data_type(io_dtype));
    auto fc1_bias =
        graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("fc1_bias")
                          .set_uid(FC1_BIAS_UID)
                          .set_dim({batch, args.hidden_dim, 1})
                          .set_stride(channel_bias_stride(args.row_major, args.hidden_dim))
                          .set_data_type(fe::DataType_t::FLOAT));

    auto fc1 = graph->matmul(fc1_w,
                             input,
                             fe::graph::Matmul_attributes().set_name("fc1").set_compute_data_type(
                                 fe::DataType_t::FLOAT));
    fc1->set_data_type(fe::DataType_t::FLOAT);

    auto fc1_biased = graph->pointwise(fc1,
                                       fc1_bias,
                                       fe::graph::Pointwise_attributes()
                                           .set_name("fc1_bias_add")
                                           .set_mode(fe::PointwiseMode_t::ADD)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
    fc1_biased->set_data_type(fe::DataType_t::FLOAT);

    auto gelu =
        graph->pointwise(fc1_biased,
                         fe::graph::Pointwise_attributes()
                             .set_name("fc1_gelu")
                             .set_mode(args.approx_gelu ? fe::PointwiseMode_t::GELU_APPROX_TANH_FWD
                                                        : fe::PointwiseMode_t::GELU_FWD)
                             .set_compute_data_type(fe::DataType_t::FLOAT));
    gelu->set_dim({batch, args.hidden_dim, args.cols})
        .set_stride(matrix_stride(args.row_major, args.hidden_dim, args.cols))
        .set_data_type(fc1_out_dtype);

    if (!args.full_chain) {
        gelu->set_output(true).set_uid(FC1_OUT_UID);
        return graph;
    }

    auto fc2_w = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("fc2_w")
            .set_uid(FC2_W_UID)
            .set_dim({batch, args.output_dim, args.hidden_dim})
            .set_stride(weight_stride(
                args.row_major, args.ggml_weight_layout, args.output_dim, args.hidden_dim))
            .set_data_type(io_dtype));
    auto fc2_bias =
        graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("fc2_bias")
                          .set_uid(FC2_BIAS_UID)
                          .set_dim({batch, args.output_dim, 1})
                          .set_stride(channel_bias_stride(args.row_major, args.output_dim))
                          .set_data_type(fe::DataType_t::FLOAT));

    auto fc2 = graph->matmul(fc2_w,
                             gelu,
                             fe::graph::Matmul_attributes().set_name("fc2").set_compute_data_type(
                                 fe::DataType_t::FLOAT));
    fc2->set_data_type(fe::DataType_t::FLOAT);

    auto output = graph->pointwise(fc2,
                                   fc2_bias,
                                   fe::graph::Pointwise_attributes()
                                       .set_name("fc2_bias_add")
                                       .set_mode(fe::PointwiseMode_t::ADD)
                                       .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_output(true)
        .set_uid(FINAL_OUT_UID)
        .set_dim({batch, args.output_dim, args.cols})
        .set_stride(matrix_stride(args.row_major, args.output_dim, args.cols))
        .set_data_type(final_out_dtype);

    return graph;
}

struct EventPair {
    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    ~EventPair() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

double percentile(std::vector<float> values, double q) {
    std::ranges::sort(values);
    const double index = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(index);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = index - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - frac) + static_cast<double>(values[hi]) * frac;
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args = parse_args(argc, argv);

    cudnnHandle_t handle = nullptr;
    CUDNN_CHECK(cudnnCreate(&handle));
    std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, decltype(&cudnnDestroy)> handle_guard(
        handle, cudnnDestroy);

    auto graph = create_mlp_graph(args);
    auto status = graph->build(handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        throw std::runtime_error(
            std::format("cuDNN MLP graph build failed: {}", status.get_message()));
    }

    const size_t io_bytes = dtype_size(args.io_dtype);
    const size_t fc1_out_bytes = dtype_size(args.fc1_out_dtype);
    const size_t final_out_bytes = dtype_size(args.final_out_dtype);
    const size_t fc1_weight_elems = static_cast<size_t>(args.hidden_dim * args.input_dim);
    const size_t input_elems = static_cast<size_t>(args.input_dim * args.cols);
    const size_t fc1_bias_elems = static_cast<size_t>(args.hidden_dim);
    const size_t fc1_out_elems = static_cast<size_t>(args.hidden_dim * args.cols);
    const size_t fc2_weight_elems = static_cast<size_t>(args.output_dim * args.hidden_dim);
    const size_t fc2_bias_elems = static_cast<size_t>(args.output_dim);
    const size_t final_out_elems = static_cast<size_t>(args.output_dim * args.cols);

    DeviceBuffer<std::byte> fc1_w(fc1_weight_elems * io_bytes);
    DeviceBuffer<std::byte> input(input_elems * io_bytes);
    DeviceBuffer<float> fc1_bias(fc1_bias_elems);
    DeviceBuffer<std::byte> fc1_out(args.full_chain ? 0 : fc1_out_elems * fc1_out_bytes);
    DeviceBuffer<std::byte> fc2_w(args.full_chain ? fc2_weight_elems * io_bytes : 0);
    DeviceBuffer<float> fc2_bias(args.full_chain ? fc2_bias_elems : 0);
    DeviceBuffer<std::byte> final_out(args.full_chain ? final_out_elems * final_out_bytes : 0);

    std::vector<uint16_t> fc1_w_host_bf16;
    std::vector<uint16_t> input_host_bf16;
    std::vector<float> fc1_bias_host;
    if (args.check) {
        if (args.full_chain || args.io_dtype != "bf16" ||
            (args.fc1_out_dtype != "bf16" && args.fc1_out_dtype != "fp32")) {
            throw std::runtime_error(
                "--check currently supports fc1-only bf16 input/output checks");
        }
        const int64_t ref_ops = args.input_dim * args.hidden_dim * args.cols;
        if (ref_ops > 20'000'000) {
            throw std::runtime_error("--check is intended for small shapes; reduce dimensions");
        }
        fc1_w_host_bf16 = make_bf16_pattern(fc1_weight_elems, 31, 1.0f / 32.0f);
        input_host_bf16 = make_bf16_pattern(input_elems, 29, 1.0f / 32.0f);
        fc1_bias_host = make_f32_pattern(fc1_bias_elems, 23, 1.0f / 64.0f);
        CUDA_CHECK(
            cudaMemcpy(fc1_w.get(), fc1_w_host_bf16.data(), fc1_w.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(input.get(), input_host_bf16.data(), input.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            fc1_bias.get(), fc1_bias_host.data(), fc1_bias.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(fc1_out.get(), 0, fc1_out.bytes()));
    } else {
        CUDA_CHECK(cudaMemset(fc1_w.get(), 0, fc1_w.bytes()));
        CUDA_CHECK(cudaMemset(input.get(), 0, input.bytes()));
        CUDA_CHECK(cudaMemset(fc1_bias.get(), 0, fc1_bias.bytes()));
        if (!args.full_chain) {
            CUDA_CHECK(cudaMemset(fc1_out.get(), 0, fc1_out.bytes()));
        } else {
            CUDA_CHECK(cudaMemset(fc2_w.get(), 0, fc2_w.bytes()));
            CUDA_CHECK(cudaMemset(fc2_bias.get(), 0, fc2_bias.bytes()));
            CUDA_CHECK(cudaMemset(final_out.get(), 0, final_out.bytes()));
        }
    }

    int64_t workspace_size = 0;
    status = graph->get_workspace_size(workspace_size);
    if (!status.is_good()) {
        throw std::runtime_error(std::format("workspace query failed: {}", status.get_message()));
    }
    DeviceBuffer<std::byte> workspace(static_cast<size_t>(workspace_size));

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {FC1_W_UID, fc1_w.get()},
        {INPUT_UID, input.get()},
        {FC1_BIAS_UID, fc1_bias.get()},
    };
    if (args.full_chain) {
        variant_pack[FC2_W_UID] = fc2_w.get();
        variant_pack[FC2_BIAS_UID] = fc2_bias.get();
        variant_pack[FINAL_OUT_UID] = final_out.get();
    } else {
        variant_pack[FC1_OUT_UID] = fc1_out.get();
    }

    auto run_once = [&] {
        auto execute_status = graph->execute(handle, variant_pack, workspace.get());
        if (!execute_status.is_good()) {
            throw std::runtime_error(
                std::format("execute failed: {}", execute_status.get_message()));
        }
    };

    for (int i = 0; i < args.warmup; ++i) {
        run_once();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    EventPair events;
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(args.iters));
    for (int i = 0; i < args.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(events.start));
        run_once();
        CUDA_CHECK(cudaEventRecord(events.stop));
        CUDA_CHECK(cudaEventSynchronize(events.stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop));
        samples.push_back(elapsed_ms);
    }

    const double mean_ms =
        std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    const double median_ms = percentile(samples, 0.5);
    const double p95_ms = percentile(samples, 0.95);
    const auto [min_it, max_it] = std::ranges::minmax_element(samples);

    int64_t check_bad = 0;
    float check_max_abs = 0.0f;
    float check_max_rel = 0.0f;
    if (args.check || !args.dump_output.empty()) {
        if (args.full_chain) {
            throw std::runtime_error("--check/--dump-output currently supports fc1-only");
        }

        std::vector<float> actual_f32;
        std::vector<uint16_t> actual_bf16;
        if (args.fc1_out_dtype == "fp32") {
            actual_f32.resize(fc1_out_elems);
            CUDA_CHECK(cudaMemcpy(
                actual_f32.data(), fc1_out.get(), fc1_out.bytes(), cudaMemcpyDeviceToHost));
        } else if (args.fc1_out_dtype == "bf16") {
            actual_bf16.resize(fc1_out_elems);
            CUDA_CHECK(cudaMemcpy(
                actual_bf16.data(), fc1_out.get(), fc1_out.bytes(), cudaMemcpyDeviceToHost));
        } else {
            throw std::runtime_error(
                "--check/--dump-output currently supports bf16/fp32 fc1 outputs");
        }

        if (!args.dump_output.empty()) {
            std::ofstream out(args.dump_output, std::ios::binary);
            if (!out) {
                throw std::runtime_error(
                    std::format("open dump output failed: {}", args.dump_output));
            }
            if (args.fc1_out_dtype == "fp32") {
                out.write(reinterpret_cast<const char*>(actual_f32.data()),
                          static_cast<std::streamsize>(actual_f32.size() * sizeof(float)));
            } else {
                out.write(reinterpret_cast<const char*>(actual_bf16.data()),
                          static_cast<std::streamsize>(actual_bf16.size() * sizeof(uint16_t)));
            }
            if (!out) {
                throw std::runtime_error(
                    std::format("write dump output failed: {}", args.dump_output));
            }
        }

        if (args.check) {
            auto weight_at = [&](int64_t row, int64_t col) -> float {
                const size_t index = (args.row_major || args.ggml_weight_layout)
                                         ? static_cast<size_t>(row * args.input_dim + col)
                                         : static_cast<size_t>(row + col * args.hidden_dim);
                return bf16_bits_to_f32(fc1_w_host_bf16[index]);
            };
            auto input_at = [&](int64_t row, int64_t col) -> float {
                const size_t index = args.row_major
                                         ? static_cast<size_t>(row * args.cols + col)
                                         : static_cast<size_t>(row + col * args.input_dim);
                return bf16_bits_to_f32(input_host_bf16[index]);
            };

            for (int64_t row = 0; row < args.hidden_dim; ++row) {
                for (int64_t col = 0; col < args.cols; ++col) {
                    float acc = fc1_bias_host[static_cast<size_t>(row)];
                    for (int64_t k = 0; k < args.input_dim; ++k) {
                        acc += weight_at(row, k) * input_at(k, col);
                    }
                    float expected = gelu(acc, args.approx_gelu);
                    const size_t out_index = args.row_major
                                                 ? static_cast<size_t>(row * args.cols + col)
                                                 : static_cast<size_t>(row + col * args.hidden_dim);
                    float actual = 0.0f;
                    if (args.fc1_out_dtype == "fp32") {
                        actual = actual_f32[out_index];
                    } else {
                        expected = bf16_bits_to_f32(f32_to_bf16_bits(expected));
                        actual = bf16_bits_to_f32(actual_bf16[out_index]);
                    }
                    const float abs_delta = std::abs(actual - expected);
                    const float rel_delta = abs_delta / std::max(std::abs(expected), 1e-6f);
                    check_max_abs = std::max(check_max_abs, abs_delta);
                    check_max_rel = std::max(check_max_rel, rel_delta);
                    if (abs_delta > args.tolerance && rel_delta > args.tolerance) {
                        ++check_bad;
                    }
                }
            }
        }
    }

    std::cout << std::format(
        "backend=cudnn-mlp mode={} layout={} weight_layout={} io_dtype={} fc1_out_dtype={} "
        "final_out_dtype={} "
        "input_dim={} hidden_dim={} output_dim={} cols={} approx_gelu={} warmup={} iters={} "
        "mean_ms={:.6f} median_ms={:.6f} p95_ms={:.6f} min_ms={:.6f} max_ms={:.6f} "
        "workspace_bytes={} check={} check_bad={} check_max_abs={:.9g} check_max_rel={:.9g}\n",
        args.full_chain ? "full_chain" : "fc1_gelu",
        args.row_major ? "row_major" : "col_major",
        args.ggml_weight_layout ? "ggml_transposed" : "native",
        args.io_dtype,
        args.fc1_out_dtype,
        args.final_out_dtype,
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.cols,
        args.approx_gelu ? 1 : 0,
        args.warmup,
        args.iters,
        mean_ms,
        median_ms,
        p95_ms,
        *min_it,
        *max_it,
        workspace_size,
        args.check ? 1 : 0,
        check_bad,
        check_max_abs,
        check_max_rel);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "cudnn_mlp_bench: " << e.what() << '\n';
    return 1;
}
