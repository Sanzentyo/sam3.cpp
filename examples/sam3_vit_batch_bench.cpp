#include "sam3.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int max_block_stage = static_cast<int>(SAM3_VIT_BLOCK_STAGE_MLP_FC1_GELU);

struct Args {
    std::string model_path;
    std::filesystem::path image_path =
        "outputs/model-matrix-sam3-bf16-cached-frame2-internal-20260612a/frames/00001.jpg";
    int batch_size = 1;
    int vit_blocks = -1;
    int block_stage_index = 0;
    std::optional<sam3_vit_block_stage> block_stage;
    int warmup_runs = 2;
    int repeats = 5;
    int threads = 4;
    bool use_gpu = true;
    bool include_tracker_neck = false;
};

struct TimingStats {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double sd_ms = 0.0;
    double ci95_ms = 0.0;
};

[[nodiscard]] int parse_int(std::string_view text, std::string_view name, int min_value) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value < min_value) {
        throw std::invalid_argument(std::format("invalid {}", name));
    }
    return value;
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
        } else if (arg == "--image") {
            args.image_path = value(arg);
        } else if (arg == "--batch-size") {
            args.batch_size = parse_int(value(arg), arg, 1);
        } else if (arg == "--vit-blocks") {
            args.vit_blocks = parse_int(value(arg), arg, 0);
        } else if (arg == "--block-stage-index") {
            args.block_stage_index = parse_int(value(arg), arg, 0);
        } else if (arg == "--block0-stage" || arg == "--block-stage") {
            const int stage = parse_int(value(arg), arg, 0);
            if (stage > max_block_stage) {
                throw std::invalid_argument(std::format("invalid {}", arg));
            }
            args.block_stage = static_cast<sam3_vit_block_stage>(stage);
            if (arg == "--block0-stage") {
                args.block_stage_index = 0;
            }
        } else if (arg == "--warmup-runs") {
            args.warmup_runs = parse_int(value(arg), arg, 0);
        } else if (arg == "--repeats") {
            args.repeats = parse_int(value(arg), arg, 1);
        } else if (arg == "--threads") {
            args.threads = parse_int(value(arg), arg, 1);
        } else if (arg == "--gpu") {
            args.use_gpu = true;
        } else if (arg == "--cpu") {
            args.use_gpu = false;
        } else if (arg == "--with-tracker-neck") {
            args.include_tracker_neck = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << std::format(
                "Usage: {} --model <path> [--image <path>] [--batch-size <n>] "
                "[--vit-blocks <n>] [--block-stage-index <n>] "
                "[--block-stage <0..{}>|--block0-stage <0..{}>] [--with-tracker-neck] "
                "[--warmup-runs <n>] "
                "[--repeats <n>] [--gpu|--cpu]\n",
                argv[0],
                max_block_stage,
                max_block_stage);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::format("unknown argument {}", arg));
        }
    }
    if (args.model_path.empty()) {
        throw std::invalid_argument("--model is required");
    }
    return args;
}

[[nodiscard]] TimingStats summarize(std::vector<double> values) {
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
        double squared_error_sum = 0.0;
        for (const double value : values) {
            const double delta = value - total / static_cast<double>(values.size());
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

[[nodiscard]] std::string format_samples_json(const std::vector<double>& values, double divisor) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::format("{:.6f}", values[i] / divisor);
    }
    out += "]";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        sam3_params params;
        params.model_path = args.model_path;
        params.use_gpu = args.use_gpu;
        params.require_gpu = args.use_gpu;
        params.n_threads = args.threads;

        auto model = sam3_load_model(params);
        if (!model) {
            std::cerr << "failed to load model\n";
            return 1;
        }

        sam3_image image = sam3_load_image(args.image_path.string());
        if (image.data.empty()) {
            std::cerr << std::format("failed to load image {}\n", args.image_path.string());
            return 1;
        }

        std::vector<double> timings;
        int64_t output_ne[4] = {};
        std::optional<sam3_batch_lane_check> lane_check;
        if (!sam3_test_bench_sam3_vit_batch(*model,
                                            image,
                                            args.batch_size,
                                            args.include_tracker_neck,
                                            args.vit_blocks,
                                            args.block_stage_index,
                                            args.block_stage,
                                            args.warmup_runs,
                                            args.repeats,
                                            timings,
                                            output_ne,
                                            lane_check,
                                            args.threads)) {
            std::cerr << "batch bench failed\n";
            return 1;
        }

        const TimingStats stats = summarize(timings);
        const auto samples_json = format_samples_json(timings, 1.0);
        const auto samples_per_frame_json =
            format_samples_json(timings, static_cast<double>(args.batch_size));
        std::cout << std::format(
            "{{\"model\":\"{}\",\"backend\":\"{}\",\"image\":\"{}\",\"batch_size\":{},"
            "\"include_tracker_neck\":{},\"vit_blocks\":{},\"block_stage_index\":{},"
            "\"block_stage\":{},\"block0_stage\":{},"
            "\"warmup_runs\":{},\"repeats\":{},"
            "\"mean_ms\":{:.6f},\"median_ms\":{:.6f},\"min_ms\":{:.6f},"
            "\"max_ms\":{:.6f},\"sd_ms\":{:.6f},\"ci95_ms\":{:.6f},"
            "\"mean_ms_per_frame\":{:.6f},\"sd_ms_per_frame\":{:.6f},"
            "\"ci95_ms_per_frame\":{:.6f},\"median_ms_per_frame\":{:.6f},"
            "\"min_ms_per_frame\":{:.6f},\"max_ms_per_frame\":{:.6f},"
            "\"samples_ms\":{},\"samples_ms_per_frame\":{},"
            "\"output_ne\":[{},{},{},{}],\"lane_max_abs_diff\":{},"
            "\"lane_mean_abs_diff\":{},\"lane_compared_lanes\":{}}}\n",
            args.model_path,
            sam3_backend_name(*model),
            args.image_path.string(),
            args.batch_size,
            args.include_tracker_neck,
            args.vit_blocks,
            args.block_stage_index,
            args.block_stage ? std::to_string(static_cast<int>(*args.block_stage)) : "null",
            args.block_stage && args.block_stage_index == 0
                ? std::to_string(static_cast<int>(*args.block_stage))
                : "null",
            args.warmup_runs,
            args.repeats,
            stats.mean_ms,
            stats.median_ms,
            stats.min_ms,
            stats.max_ms,
            stats.sd_ms,
            stats.ci95_ms,
            stats.mean_ms / static_cast<double>(args.batch_size),
            stats.sd_ms / static_cast<double>(args.batch_size),
            stats.ci95_ms / static_cast<double>(args.batch_size),
            stats.median_ms / static_cast<double>(args.batch_size),
            stats.min_ms / static_cast<double>(args.batch_size),
            stats.max_ms / static_cast<double>(args.batch_size),
            samples_json,
            samples_per_frame_json,
            output_ne[0],
            output_ne[1],
            output_ne[2],
            output_ne[3],
            lane_check ? std::format("{:.9g}", lane_check->max_abs_diff) : "null",
            lane_check ? std::format("{:.9g}", lane_check->mean_abs_diff) : "null",
            lane_check ? std::to_string(lane_check->compared_lanes) : "null");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam3_vit_batch_bench failed: {}\n", ex.what());
        return 1;
    }
}
