#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

extern "C" cudaError_t sam3_cudnn_sdpa_convert_f32_to_dtype(const float* src,
                                                            void* dst,
                                                            size_t count,
                                                            int dtype_id);

namespace {

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t STATS_UID = 5;

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
    size_t count() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

std::shared_ptr<fe::graph::Graph> create_sdpa_forward_graph(int64_t batch,
                                                            int64_t heads,
                                                            int64_t q_seq,
                                                            int64_t kv_seq,
                                                            int64_t head_dim,
                                                            bool generate_stats,
                                                            fe::DataType_t io_dtype,
                                                            fe::DataType_t out_dtype) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const std::vector<int64_t> q_dims = {batch, heads, q_seq, head_dim};
    const std::vector<int64_t> kv_dims = {batch, heads, kv_seq, head_dim};
    const std::vector<int64_t> q_stride = {heads * q_seq * head_dim, q_seq * head_dim, head_dim, 1};
    const std::vector<int64_t> kv_stride = {
        heads * kv_seq * head_dim, kv_seq * head_dim, head_dim, 1};

    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim(q_dims)
                               .set_stride(q_stride)
                               .set_data_type(io_dtype));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim(kv_dims)
                               .set_stride(kv_stride)
                               .set_data_type(io_dtype));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim(kv_dims)
                               .set_stride(kv_stride)
                               .set_data_type(io_dtype));

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("sam3_head64_sdpa")
                            .set_generate_stats(generate_stats)
                            .set_attn_scale(attn_scale);

    auto [o, stats] = graph->sdpa(q, k, v, sdpa_options);
    o->set_output(true)
        .set_dim(q_dims)
        .set_stride({heads * head_dim, head_dim, batch * heads * head_dim, 1})
        .set_data_type(out_dtype)
        .set_uid(O_UID);

    if (generate_stats) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(STATS_UID);
    }

    return graph;
}

struct Args {
    int64_t batch = 1;
    int64_t heads = 16;
    int64_t seq = 5184;
    int64_t q_seq = 0;
    int64_t kv_seq = 0;
    int64_t head_dim = 64;
    int warmup = 5;
    int iters = 30;
    bool generate_stats = false;
    bool convert_f32_inputs = false;
    std::string io_dtype = "bf16";
    std::string out_dtype = "bf16";
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

int dtype_id(std::string_view value) {
    if (value == "bf16") {
        return 0;
    }
    if (value == "fp16") {
        return 1;
    }
    if (value == "fp32") {
        return 2;
    }
    throw std::runtime_error(std::format("unsupported dtype: {}", value));
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::format("{} needs a value", name));
            }
            return argv[++i];
        };
        if (key == "--batch") {
            args.batch = std::stoll(need_value("--batch"));
        } else if (key == "--heads") {
            args.heads = std::stoll(need_value("--heads"));
        } else if (key == "--seq") {
            args.seq = std::stoll(need_value("--seq"));
        } else if (key == "--q-seq") {
            args.q_seq = std::stoll(need_value("--q-seq"));
        } else if (key == "--kv-seq") {
            args.kv_seq = std::stoll(need_value("--kv-seq"));
        } else if (key == "--head-dim") {
            args.head_dim = std::stoll(need_value("--head-dim"));
        } else if (key == "--warmup") {
            args.warmup = std::stoi(need_value("--warmup"));
        } else if (key == "--iters") {
            args.iters = std::stoi(need_value("--iters"));
        } else if (key == "--stats") {
            args.generate_stats = true;
        } else if (key == "--convert-f32-inputs") {
            args.convert_f32_inputs = true;
        } else if (key == "--io-dtype") {
            args.io_dtype = need_value("--io-dtype");
            parse_dtype(args.io_dtype);
        } else if (key == "--out-dtype") {
            args.out_dtype = need_value("--out-dtype");
            parse_dtype(args.out_dtype);
        } else {
            throw std::runtime_error(std::format("unknown argument: {}", key));
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    if (args.q_seq <= 0) {
        args.q_seq = args.seq;
    }
    if (args.kv_seq <= 0) {
        args.kv_seq = args.seq;
    }

    cudnnHandle_t handle = nullptr;
    CUDNN_CHECK(cudnnCreate(&handle));
    std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, decltype(&cudnnDestroy)> handle_guard(
        handle, cudnnDestroy);

    auto graph = create_sdpa_forward_graph(args.batch,
                                           args.heads,
                                           args.q_seq,
                                           args.kv_seq,
                                           args.head_dim,
                                           args.generate_stats,
                                           parse_dtype(args.io_dtype),
                                           parse_dtype(args.out_dtype));
    auto status = graph->build(handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        throw std::runtime_error(
            std::format("cuDNN SDPA graph build failed: {}", status.get_message()));
    }

    const size_t q_count =
        static_cast<size_t>(args.batch * args.heads * args.q_seq * args.head_dim);
    const size_t kv_count =
        static_cast<size_t>(args.batch * args.heads * args.kv_seq * args.head_dim);
    DeviceBuffer<std::byte> q(q_count * dtype_size(args.io_dtype));
    DeviceBuffer<std::byte> k(kv_count * dtype_size(args.io_dtype));
    DeviceBuffer<std::byte> v(kv_count * dtype_size(args.io_dtype));
    DeviceBuffer<std::byte> o(q_count * dtype_size(args.out_dtype));
    DeviceBuffer<float> q_f32(args.convert_f32_inputs ? q_count : 0);
    DeviceBuffer<float> k_f32(args.convert_f32_inputs ? kv_count : 0);
    DeviceBuffer<float> v_f32(args.convert_f32_inputs ? kv_count : 0);
    DeviceBuffer<float> stats(
        args.generate_stats ? static_cast<size_t>(args.batch * args.heads * args.q_seq) : 0);

    if (args.convert_f32_inputs) {
        CUDA_CHECK(cudaMemset(q_f32.get(), 0, q_f32.count() * sizeof(float)));
        CUDA_CHECK(cudaMemset(k_f32.get(), 0, k_f32.count() * sizeof(float)));
        CUDA_CHECK(cudaMemset(v_f32.get(), 0, v_f32.count() * sizeof(float)));
    }

    int64_t workspace_size = 0;
    status = graph->get_workspace_size(workspace_size);
    if (!status.is_good()) {
        throw std::runtime_error(std::format("workspace query failed: {}", status.get_message()));
    }
    DeviceBuffer<std::byte> workspace(static_cast<size_t>(workspace_size));

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, q.get()},
        {K_UID, k.get()},
        {V_UID, v.get()},
        {O_UID, o.get()},
    };
    if (args.generate_stats) {
        variant_pack[STATS_UID] = stats.get();
    }

    auto run_once = [&] {
        if (args.convert_f32_inputs) {
            const int id = dtype_id(args.io_dtype);
            CUDA_CHECK(sam3_cudnn_sdpa_convert_f32_to_dtype(q_f32.get(), q.get(), q_count, id));
            CUDA_CHECK(sam3_cudnn_sdpa_convert_f32_to_dtype(k_f32.get(), k.get(), kv_count, id));
            CUDA_CHECK(sam3_cudnn_sdpa_convert_f32_to_dtype(v_f32.get(), v.get(), kv_count, id));
        }
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

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < args.iters; ++i) {
        run_once();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << std::format(
        "backend=cudnn-sdpa io_dtype={} out_dtype={} batch={} heads={} seq={} head_dim={} "
        "q_seq={} kv_seq={} stats={} convert_f32_inputs={} mean_ms={:.6f} workspace_bytes={}\n",
        args.io_dtype,
        args.out_dtype,
        args.batch,
        args.heads,
        args.seq,
        args.head_dim,
        args.q_seq,
        args.kv_seq,
        args.generate_stats ? 1 : 0,
        args.convert_f32_inputs ? 1 : 0,
        elapsed_ms / static_cast<float>(args.iters),
        workspace_size);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "cudnn_sdpa_bench: " << e.what() << '\n';
    return 1;
}
