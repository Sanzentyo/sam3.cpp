#include "sam3.h"

#include <array>
#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::filesystem::path input_prefix =
        "outputs/sam31-mask-init-state-python/expected_cpp_layout/input_image_preprocessed";
    std::filesystem::path output_prefix = "outputs/sam31-vit-prefix-cpp/vit_patch_embed";
    sam3_vit_prefix_stage stage = SAM3_VIT_PREFIX_STAGE_PATCH_EMBED;
    int threads = 4;
    bool use_gpu = false;
};

[[nodiscard]] int parse_positive_int(std::string_view text, std::string_view name) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value <= 0) {
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
        } else if (arg == "--input-prefix") {
            args.input_prefix = value(arg);
        } else if (arg == "--out-prefix") {
            args.output_prefix = value(arg);
        } else if (arg == "--stage") {
            const auto stage = value(arg);
            if (stage == "patch_embed") {
                args.stage = SAM3_VIT_PREFIX_STAGE_PATCH_EMBED;
            } else if (stage == "pos_add") {
                args.stage = SAM3_VIT_PREFIX_STAGE_POS_ADD;
            } else if (stage == "ln_pre") {
                args.stage = SAM3_VIT_PREFIX_STAGE_LN_PRE;
            } else {
                throw std::invalid_argument(std::format("unknown stage {}", stage));
            }
        } else if (arg == "--threads") {
            args.threads = parse_positive_int(value(arg), arg);
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
    std::array<int64_t, 4> shape{};
    for (auto& dim : shape) {
        if (!(in >> dim)) {
            throw std::runtime_error(std::format("invalid shape file {}", path.string()));
        }
    }
    return shape;
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

[[nodiscard]] size_t element_count(const std::array<int64_t, 4>& shape) {
    return static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]) *
           static_cast<size_t>(shape[2]) * static_cast<size_t>(shape[3]);
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
        if (sam3_get_model_type(*model) != SAM3_MODEL_SAM3_1) {
            std::cerr << "expected SAM3.1 model\n";
            return 1;
        }

        const auto input_shape = read_shape(args.input_prefix.string() + ".shape");
        const auto input =
            read_f32(args.input_prefix.string() + ".bin", element_count(input_shape));
        std::vector<float> output;
        std::array<int64_t, 4> output_shape{};
        if (!sam3_test_run_vit_prefix_stage(*model,
                                            args.stage,
                                            input,
                                            input_shape,
                                            output,
                                            output_shape.data(),
                                            args.threads)) {
            std::cerr << "patch embed stage failed\n";
            return 1;
        }

        write_f32_tensor(args.output_prefix, output, output_shape);
        std::cout << std::format(
            "wrote {} using backend {}\n", args.output_prefix.string(), sam3_backend_name(*model));
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam31_vit_prefix_case failed: {}\n", ex.what());
        return 1;
    }
    return 0;
}
