#include "sam3.h"

#include <charconv>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::string model_path;
    std::string case_dir = "outputs/sam31-propagation-features-case";
    std::string output_dir = "outputs/sam31-propagation-features-cpp";
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
        } else if (arg == "--case") {
            args.case_dir = value(arg);
        } else if (arg == "--out") {
            args.output_dir = value(arg);
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
            std::cerr << "failed to load SAM3.1 model\n";
            return 1;
        }
        if (sam3_get_model_type(*model) != SAM3_MODEL_SAM3_1) {
            std::cerr << "expected SAM3.1 model\n";
            return 1;
        }

        if (!sam3_test_dump_sam31_propagation_features_case(
                *model, args.case_dir, args.output_dir, args.threads)) {
            std::cerr << "SAM3.1 propagation feature case failed\n";
            return 1;
        }

        std::cout << std::format("wrote SAM3.1 propagation features to {}\n", args.output_dir);
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam31_propagation_features_case failed: {}\n", ex.what());
        return 1;
    }
    return 0;
}
