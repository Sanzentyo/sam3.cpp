#include "sam3.h"

#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::string prompt = "person";
    std::vector<int32_t> token_ids;
    std::filesystem::path output_dir = "outputs/sam3-text-encoder-cpp";
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

[[nodiscard]] std::vector<int32_t> parse_token_ids(std::string_view text) {
    std::vector<int32_t> values;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto part = text.substr(0, comma);
        int value = 0;
        const auto [ptr, ec] = std::from_chars(part.data(), part.data() + part.size(), value);
        if (ec != std::errc{} || ptr != part.data() + part.size()) {
            throw std::invalid_argument("invalid --token-ids");
        }
        values.push_back(value);
        if (comma == std::string_view::npos) {
            break;
        }
        text.remove_prefix(comma + 1);
    }
    return values;
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
        } else if (arg == "--prompt") {
            args.prompt = value(arg);
        } else if (arg == "--token-ids") {
            args.token_ids = parse_token_ids(value(arg));
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

bool dump_token_ids(const std::filesystem::path& output_dir,
                    const std::vector<int32_t>& token_ids) {
    {
        std::ofstream out(output_dir / "token_ids.bin", std::ios::binary);
        if (!out) {
            return false;
        }
        out.write(reinterpret_cast<const char*>(token_ids.data()),
                  static_cast<std::streamsize>(token_ids.size() * sizeof(int32_t)));
    }
    {
        std::ofstream out(output_dir / "token_ids.shape");
        if (!out) {
            return false;
        }
        out << token_ids.size() << "\n";
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        std::filesystem::create_directories(args.output_dir);

        sam3_params params;
        params.model_path = args.model_path;
        params.n_threads = args.threads;
        params.use_gpu = args.use_gpu;
        params.require_gpu = args.use_gpu;

        auto model = sam3_load_model(params);
        if (!model) {
            std::cerr << "failed to load SAM3 model\n";
            return 1;
        }
        if (sam3_get_model_type(*model) != SAM3_MODEL_SAM3) {
            std::cerr << std::format("expected SAM3 model, got backend {}\n",
                                     sam3_backend_name(*model));
            return 1;
        }
        auto token_ids = args.token_ids;
        if (token_ids.empty()) {
            if (!sam3_test_load_tokenizer(args.model_path)) {
                std::cerr << "failed to load tokenizer\n";
                return 1;
            }
            token_ids = sam3_test_tokenize(args.prompt);
        }
        if (token_ids.empty()) {
            std::cerr << "failed to tokenize prompt\n";
            return 1;
        }
        if (!dump_token_ids(args.output_dir, token_ids)) {
            std::cerr << "failed to dump token IDs\n";
            return 1;
        }
        if (!sam3_test_dump_text_encoder(
                *model, token_ids, args.output_dir.string(), args.threads)) {
            std::cerr << "text encoder dump failed\n";
            return 1;
        }
        std::cout << std::format("wrote SAM3 text encoder outputs to {}\n",
                                 args.output_dir.string());
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam3_text_encoder_case failed: {}\n", ex.what());
        return 1;
    }
    return 0;
}
