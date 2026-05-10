#include "sam3.h"

#include <charconv>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

struct smoke_args {
    std::string model_path;
    std::optional<std::string> image_path;
    std::optional<std::string> video_path;
    std::optional<std::string> text_prompt;
    std::optional<std::string> output_dir;
    int frame_index = 0;
    int n_threads = 4;
    bool use_gpu = true;
    bool multimask = false;
    std::optional<float> point_x;
    std::optional<float> point_y;
};

[[nodiscard]] std::optional<int> parse_int(std::string_view value) {
    int parsed = 0;
    const auto* const begin = value.data();
    const auto* const end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::optional<float> parse_float(std::string_view value) {
    float parsed = 0.0f;
    const auto* const begin = value.data();
    const auto* const end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
        return std::nullopt;
    }
    return parsed;
}

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --model <path> (--image <path> | --video <path> [--frame <n>]) "
                 "[--point-x <x> --point-y <y>] [--text <prompt>] [--cpu] "
                 "[--threads <n>] [--multimask] [--output-dir <dir>]\n";
}

[[nodiscard]] std::optional<smoke_args> parse_args(int argc, char** argv) {
    smoke_args args{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        auto next = [&]() -> std::optional<std::string_view> {
            if (i + 1 >= argc) {
                return std::nullopt;
            }
            return std::string_view{argv[++i]};
        };

        if (arg == "--model") {
            if (const auto value = next()) {
                args.model_path = *value;
            } else {
                return std::nullopt;
            }
        } else if (arg == "--image") {
            if (const auto value = next()) {
                args.image_path = std::string{*value};
            } else {
                return std::nullopt;
            }
        } else if (arg == "--video") {
            if (const auto value = next()) {
                args.video_path = std::string{*value};
            } else {
                return std::nullopt;
            }
        } else if (arg == "--frame") {
            const auto value = next().and_then(parse_int);
            if (!value) {
                return std::nullopt;
            }
            args.frame_index = *value;
        } else if (arg == "--threads") {
            const auto value = next().and_then(parse_int);
            if (!value || *value <= 0) {
                return std::nullopt;
            }
            args.n_threads = *value;
        } else if (arg == "--point-x") {
            const auto value = next().and_then(parse_float);
            if (!value) {
                return std::nullopt;
            }
            args.point_x = value;
        } else if (arg == "--point-y") {
            const auto value = next().and_then(parse_float);
            if (!value) {
                return std::nullopt;
            }
            args.point_y = value;
        } else if (arg == "--text") {
            if (const auto value = next()) {
                args.text_prompt = std::string{*value};
            } else {
                return std::nullopt;
            }
        } else if (arg == "--output-dir") {
            if (const auto value = next()) {
                args.output_dir = std::string{*value};
            } else {
                return std::nullopt;
            }
        } else if (arg == "--cpu") {
            args.use_gpu = false;
        } else if (arg == "--multimask") {
            args.multimask = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            return std::nullopt;
        }
    }

    if (args.model_path.empty() || (args.image_path.has_value() == args.video_path.has_value())) {
        return std::nullopt;
    }
    if (args.point_x.has_value() != args.point_y.has_value()) {
        return std::nullopt;
    }
    return args;
}

[[nodiscard]] sam3_image load_smoke_image(const smoke_args& args) {
    if (args.image_path) {
        return sam3_load_image(*args.image_path);
    }
    return sam3_decode_video_frame(*args.video_path, args.frame_index);
}

void save_result_masks(const sam3_result& result,
                       const std::filesystem::path& dir,
                       std::string_view stem) {
    std::filesystem::create_directories(dir);
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        if (detection.mask.data.empty()) {
            continue;
        }
        const auto path = dir / std::format("{}_{:02}.png", stem, i);
        (void) sam3_save_mask(detection.mask, path.string());
    }
}

void print_result(std::string_view name, const sam3_result& result) {
    std::cout << std::format("{}_detections={}\n", name, result.detections.size());
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        std::cout << std::format(
            "{}_{} score={:.6f} iou={:.6f} box=[{:.1f},{:.1f},{:.1f},{:.1f}] "
            "mask={}x{} bytes={}\n",
            name,
            i,
            detection.score,
            detection.iou_score,
            detection.box.x0,
            detection.box.y0,
            detection.box.x1,
            detection.box.y1,
            detection.mask.width,
            detection.mask.height,
            detection.mask.data.size());
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    if (!args) {
        print_usage(argv[0]);
        return 2;
    }

    sam3_params params{};
    params.model_path = args->model_path;
    params.n_threads = args->n_threads;
    params.use_gpu = args->use_gpu;

    auto model = sam3_load_model(params);
    if (!model) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        std::cerr << "failed to create state\n";
        return 1;
    }

    auto image = load_smoke_image(*args);
    if (image.data.empty()) {
        std::cerr << "failed to load image input\n";
        return 1;
    }

    if (!sam3_encode_image(*state, *model, image)) {
        std::cerr << "failed to encode image\n";
        return 1;
    }

    sam3_pvs_params pvs{};
    pvs.multimask = args->multimask;
    pvs.pos_points.push_back(sam3_point{
        .x = args->point_x.value_or(static_cast<float>(image.width) * 0.5f),
        .y = args->point_y.value_or(static_cast<float>(image.height) * 0.5f),
    });
    auto pvs_result = sam3_segment_pvs(*state, *model, pvs);
    print_result("pvs", pvs_result);

    if (args->output_dir) {
        save_result_masks(pvs_result, *args->output_dir, "pvs");
    }

    if (args->text_prompt && !args->text_prompt->empty()) {
        sam3_pcs_params pcs{};
        pcs.text_prompt = *args->text_prompt;
        auto pcs_result = sam3_segment_pcs(*state, *model, pcs);
        print_result("pcs", pcs_result);
        if (args->output_dir) {
            save_result_masks(pcs_result, *args->output_dir, "pcs");
        }
    }

    return pvs_result.detections.empty() ? 1 : 0;
}
