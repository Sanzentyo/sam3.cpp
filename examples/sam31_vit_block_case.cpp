#include "sam3.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::filesystem::path input_prefix =
        "outputs/sam31-mask-init-state-python/expected_cpp_layout/vit_block_00_input";
    std::filesystem::path output_prefix = "outputs/sam31-vit-block-cpp/vit_block_00_norm1";
    sam3_vit_block_stage stage = SAM3_VIT_BLOCK_STAGE_NORM1;
    int block_idx = 0;
    int threads = 4;
    int warmup_runs = 0;
    int repeats = 1;
    bool use_gpu = false;
};

struct TimingStats {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double sd_ms = 0.0;
    double ci95_ms = 0.0;
};

[[nodiscard]] int parse_positive_int(std::string_view text, std::string_view name) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value <= 0) {
        throw std::invalid_argument(std::format("invalid {}", name));
    }
    return value;
}

[[nodiscard]] int parse_nonnegative_int(std::string_view text, std::string_view name) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value < 0) {
        throw std::invalid_argument(std::format("invalid {}", name));
    }
    return value;
}

[[nodiscard]] sam3_vit_block_stage parse_stage(std::string_view stage) {
    if (stage == "norm1") {
        return SAM3_VIT_BLOCK_STAGE_NORM1;
    }
    if (stage == "window_part") {
        return SAM3_VIT_BLOCK_STAGE_WINDOW_PART;
    }
    if (stage == "qkv_proj") {
        return SAM3_VIT_BLOCK_STAGE_QKV_PROJ;
    }
    if (stage == "qkv_layout") {
        return SAM3_VIT_BLOCK_STAGE_QKV_LAYOUT;
    }
    if (stage == "qkv_rope") {
        return SAM3_VIT_BLOCK_STAGE_QKV_ROPE;
    }
    if (stage == "attn_core") {
        return SAM3_VIT_BLOCK_STAGE_ATTN_CORE;
    }
    if (stage == "attn_proj") {
        return SAM3_VIT_BLOCK_STAGE_ATTN_PROJ;
    }
    if (stage == "window_unpart") {
        return SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART;
    }
    if (stage == "norm2") {
        return SAM3_VIT_BLOCK_STAGE_NORM2;
    }
    if (stage == "mlp_fc1") {
        return SAM3_VIT_BLOCK_STAGE_MLP_FC1;
    }
    if (stage == "mlp_gelu") {
        return SAM3_VIT_BLOCK_STAGE_MLP_GELU;
    }
    if (stage == "mlp_fc1_gelu") {
        return SAM3_VIT_BLOCK_STAGE_MLP_FC1_GELU;
    }
    if (stage == "mlp_fc2") {
        return SAM3_VIT_BLOCK_STAGE_MLP_FC2;
    }
    if (stage == "mlp") {
        return SAM3_VIT_BLOCK_STAGE_MLP;
    }
    if (stage == "block") {
        return SAM3_VIT_BLOCK_STAGE_BLOCK;
    }
    throw std::invalid_argument(std::format("unknown stage {}", stage));
}

[[nodiscard]] std::string_view stage_name(sam3_vit_block_stage stage) {
    switch (stage) {
        case SAM3_VIT_BLOCK_STAGE_NORM1:
            return "norm1";
        case SAM3_VIT_BLOCK_STAGE_WINDOW_PART:
            return "window_part";
        case SAM3_VIT_BLOCK_STAGE_QKV_PROJ:
            return "qkv_proj";
        case SAM3_VIT_BLOCK_STAGE_QKV_LAYOUT:
            return "qkv_layout";
        case SAM3_VIT_BLOCK_STAGE_QKV_ROPE:
            return "qkv_rope";
        case SAM3_VIT_BLOCK_STAGE_ATTN_CORE:
            return "attn_core";
        case SAM3_VIT_BLOCK_STAGE_ATTN_PROJ:
            return "attn_proj";
        case SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART:
            return "window_unpart";
        case SAM3_VIT_BLOCK_STAGE_NORM2:
            return "norm2";
        case SAM3_VIT_BLOCK_STAGE_MLP_FC1:
            return "mlp_fc1";
        case SAM3_VIT_BLOCK_STAGE_MLP_GELU:
            return "mlp_gelu";
        case SAM3_VIT_BLOCK_STAGE_MLP_FC1_GELU:
            return "mlp_fc1_gelu";
        case SAM3_VIT_BLOCK_STAGE_MLP_FC2:
            return "mlp_fc2";
        case SAM3_VIT_BLOCK_STAGE_MLP:
            return "mlp";
        case SAM3_VIT_BLOCK_STAGE_BLOCK:
            return "block";
    }
    return "unknown";
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::format("missing value for {}", name));
            }
            return argv[++i];
        };
        if (arg == "--model") {
            args.model_path = value(arg);
        } else if (arg == "--input-prefix") {
            args.input_prefix = value(arg);
        } else if (arg == "--out-prefix") {
            args.output_prefix = value(arg);
        } else if (arg == "--stage") {
            args.stage = parse_stage(value(arg));
        } else if (arg == "--block") {
            args.block_idx = parse_nonnegative_int(value(arg), arg);
        } else if (arg == "--threads") {
            args.threads = parse_positive_int(value(arg), arg);
        } else if (arg == "--warmup-runs") {
            args.warmup_runs = parse_nonnegative_int(value(arg), arg);
        } else if (arg == "--repeats") {
            args.repeats = parse_positive_int(value(arg), arg);
        } else if (arg == "--gpu") {
            args.use_gpu = true;
        } else if (arg == "--cpu") {
            args.use_gpu = false;
        } else {
            throw std::invalid_argument(std::format("unknown argument {}", arg));
        }
    }
    if (args.model_path.empty()) {
        throw std::invalid_argument("--model is required");
    }
    return args;
}

[[nodiscard]] std::array<int64_t, 4> read_shape(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(std::format("failed to open {}", path.string()));
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::ranges::replace(text, ',', ' ');
    std::array<int64_t, 4> shape{1, 1, 1, 1};
    std::istringstream shape_in{text};
    int rank = 0;
    for (auto& dim : shape) {
        if (!(shape_in >> dim)) {
            break;
        }
        ++rank;
    }
    if (rank == 0) {
        throw std::runtime_error(std::format("invalid shape file {}", path.string()));
    }
    return shape;
}

[[nodiscard]] size_t element_count(const std::array<int64_t, 4>& shape) {
    return static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]) *
           static_cast<size_t>(shape[2]) * static_cast<size_t>(shape[3]);
}

[[nodiscard]] std::vector<float> read_f32(const std::filesystem::path& path, size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::format("failed to open {}", path.string()));
    }
    std::vector<float> values(count);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error(std::format("failed to read {}", path.string()));
    }
    return values;
}

[[nodiscard]] TimingStats summarize_timings(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    std::ranges::sort(values);
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    const size_t middle = values.size() / 2;
    const double median =
        values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
    double sd = 0.0;
    if (values.size() > 1) {
        const double mean = total / static_cast<double>(values.size());
        double squared_error_sum = 0.0;
        for (const double value : values) {
            const double delta = value - mean;
            squared_error_sum += delta * delta;
        }
        sd = std::sqrt(squared_error_sum / static_cast<double>(values.size() - 1));
    }
    return TimingStats{
        .mean_ms = total / static_cast<double>(values.size()),
        .median_ms = median,
        .min_ms = values.front(),
        .max_ms = values.back(),
        .sd_ms = sd,
        .ci95_ms =
            values.size() > 1 ? 1.96 * sd / std::sqrt(static_cast<double>(values.size())) : 0.0,
    };
}

[[nodiscard]] std::string format_samples_json(const std::vector<double>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::format("{:.6f}", values[i]);
    }
    out += "]";
    return out;
}

void write_f32_tensor(const std::filesystem::path& prefix,
                      std::span<const float> values,
                      std::span<const int64_t, 4> shape) {
    std::filesystem::create_directories(prefix.parent_path());
    {
        std::ofstream shape_out(prefix.string() + ".shape");
        if (!shape_out) {
            throw std::runtime_error(std::format("failed to open {}.shape", prefix.string()));
        }
        shape_out << std::format("{} {} {} {}\n", shape[0], shape[1], shape[2], shape[3]);
    }

    std::ofstream data_out(prefix.string() + ".bin", std::ios::binary);
    if (!data_out) {
        throw std::runtime_error(std::format("failed to open {}.bin", prefix.string()));
    }
    data_out.write(reinterpret_cast<const char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        sam3_params params;
        params.model_path = args.model_path;
        params.n_threads = args.threads;
        params.use_gpu = args.use_gpu;
        params.require_gpu = args.use_gpu;

        auto model = sam3_load_model(params);
        if (!model) {
            std::cerr << "failed to load model\n";
            return 1;
        }
        const auto model_type = sam3_get_model_type(*model);
        if (model_type != SAM3_MODEL_SAM3 && model_type != SAM3_MODEL_SAM3_1) {
            std::cerr << "expected SAM3 or SAM3.1 model\n";
            return 1;
        }

        const auto input_shape = read_shape(args.input_prefix.string() + ".shape");
        const auto input =
            read_f32(args.input_prefix.string() + ".bin", element_count(input_shape));
        std::vector<float> output;
        std::array<int64_t, 4> output_shape{};
        std::vector<double> timings;
        if (!sam3_test_bench_vit_block_stage(*model,
                                             args.block_idx,
                                             args.stage,
                                             input,
                                             input_shape,
                                             args.warmup_runs,
                                             args.repeats,
                                             timings,
                                             output,
                                             output_shape.data(),
                                             args.threads)) {
            std::cerr << "vit block stage failed\n";
            return 1;
        }

        write_f32_tensor(args.output_prefix, output, output_shape);
        const auto stats = summarize_timings(timings);
        const auto samples_json = format_samples_json(timings);
        std::cout << std::format(
            "{{\"backend\":\"{}\",\"bench_mode\":\"isolated_stage\","
            "\"block\":{},\"block_stage_index\":{},\"stage\":\"{}\","
            "\"stage_name\":\"{}\",\"block_stage\":{},"
            "\"warmup_runs\":{},\"repeats\":{},"
            "\"input_prefix\":\"{}\",\"input_ne\":[{},{},{},{}],"
            "\"mean_ms\":{:.6f},\"median_ms\":{:.6f},\"min_ms\":{:.6f},\"max_ms\":{:.6f},"
            "\"sd_ms\":{:.6f},\"ci95_ms\":{:.6f},\"samples_ms\":{},"
            "\"output_ne\":[{},{},{},{}],\"out_prefix\":\"{}\"}}\n",
            json_escape(sam3_backend_name(*model)),
            args.block_idx,
            args.block_idx,
            stage_name(args.stage),
            stage_name(args.stage),
            static_cast<int>(args.stage),
            args.warmup_runs,
            args.repeats,
            json_escape(args.input_prefix.string()),
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            stats.mean_ms,
            stats.median_ms,
            stats.min_ms,
            stats.max_ms,
            stats.sd_ms,
            stats.ci95_ms,
            samples_json,
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
            json_escape(args.output_prefix.string()));
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam31_vit_block_case failed: {}\n", ex.what());
        return 1;
    }
    return 0;
}
