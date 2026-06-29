#include "sam3.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::filesystem::path image_path =
        "outputs/e2e-required-split-sam3-bf16-e2e-process-current-r3-20260630a/frames/00001.jpg";
    std::filesystem::path out_dir = "outputs/e2e-required-vit-stage-capture";
    std::vector<int> blocks = {0};
    int threads = 4;
    bool use_gpu = true;
};

struct DumpSpec {
    std::string source;
    std::filesystem::path prefix;
};

struct StageInputSpec {
    int stage = 0;
    std::string stage_name;
    std::filesystem::path input_prefix;
};

struct BlockCapture {
    int block = 0;
    bool is_global = false;
    std::vector<DumpSpec> dumps;
    std::vector<StageInputSpec> stage_inputs;
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
    args.blocks.clear();
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
        } else if (arg == "--out-dir") {
            args.out_dir = value(arg);
        } else if (arg == "--block") {
            args.blocks.push_back(parse_int(value(arg), arg, 0));
        } else if (arg == "--threads") {
            args.threads = parse_int(value(arg), arg, 1);
        } else if (arg == "--gpu") {
            args.use_gpu = true;
        } else if (arg == "--cpu") {
            args.use_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << std::format(
                "Usage: {} --model <path> --image <path> --out-dir <dir> "
                "[--block <idx> ...] [--threads <n>] [--gpu|--cpu]\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::format("unknown argument {}", arg));
        }
    }
    if (args.blocks.empty()) {
        args.blocks.push_back(0);
    }
    if (args.model_path.empty()) {
        throw std::invalid_argument("--model is required");
    }
    return args;
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

[[nodiscard]] std::string block_source_name(int block, std::string_view suffix) {
    return std::format("sam3_vit_block_{:02d}_{}", block, suffix);
}

[[nodiscard]] std::filesystem::path block_prefix(const std::filesystem::path& out_dir,
                                                 int block,
                                                 std::string_view suffix) {
    return out_dir / std::format("vit_block_{:02d}_{}", block, suffix);
}

void add_dump(BlockCapture& capture,
              std::set<std::string>& source_names,
              std::string source,
              std::filesystem::path prefix) {
    source_names.insert(source);
    capture.dumps.push_back(DumpSpec{
        .source = std::move(source),
        .prefix = std::move(prefix),
    });
}

void add_stage(BlockCapture& capture,
               int stage,
               std::string_view name,
               const std::filesystem::path& input_prefix) {
    capture.stage_inputs.push_back(StageInputSpec{
        .stage = stage,
        .stage_name = std::string{name},
        .input_prefix = input_prefix,
    });
}

[[nodiscard]] BlockCapture make_block_capture(const std::filesystem::path& out_dir,
                                              int block,
                                              bool is_global,
                                              bool flat_mlp,
                                              std::set<std::string>& source_names) {
    BlockCapture capture{
        .block = block,
        .is_global = is_global,
    };

    const auto input_source =
        block == 0 ? std::string{"vit_block_00_input"} : block_source_name(block - 1, "out");
    const auto input_prefix = block_prefix(out_dir, block, "input");
    add_dump(capture, source_names, input_source, input_prefix);

    const auto norm1_prefix = block_prefix(out_dir, block, "norm1");
    add_dump(capture, source_names, block_source_name(block, "norm1"), norm1_prefix);

    std::optional<std::filesystem::path> window_part_prefix;
    std::optional<std::filesystem::path> window_unpart_prefix;
    const std::string attn_scope = is_global ? "global" : "window";
    if (!is_global) {
        window_part_prefix = block_prefix(out_dir, block, "window_part");
        add_dump(
            capture, source_names, block_source_name(block, "window_part"), *window_part_prefix);
    }

    const auto qkv_prefix = block_prefix(out_dir, block, "qkv");
    const auto attn_core_prefix = block_prefix(out_dir, block, "attn_core");
    const auto attn_proj_prefix = block_prefix(out_dir, block, "attn_proj");
    add_dump(capture,
             source_names,
             block_source_name(block, std::format("{}_qkv", attn_scope)),
             qkv_prefix);
    add_dump(capture,
             source_names,
             block_source_name(block, std::format("{}_attn_core", attn_scope)),
             attn_core_prefix);
    add_dump(capture,
             source_names,
             block_source_name(block, std::format("{}_proj", attn_scope)),
             attn_proj_prefix);
    if (!is_global) {
        window_unpart_prefix = block_prefix(out_dir, block, "window_unpart");
        add_dump(capture,
                 source_names,
                 block_source_name(block, "window_unpart"),
                 *window_unpart_prefix);
    }

    const auto after_attn_prefix = block_prefix(out_dir, block, "after_attn_residual");
    const auto norm2_prefix = block_prefix(out_dir, block, "norm2");
    const auto mlp_fc1_prefix = block_prefix(out_dir, block, "mlp_fc1");
    const auto mlp_gelu_prefix = block_prefix(out_dir, block, "mlp_gelu");
    const auto mlp_fc2_prefix = block_prefix(out_dir, block, "mlp_fc2");
    const auto out_prefix = block_prefix(out_dir, block, "out");
    add_dump(
        capture, source_names, block_source_name(block, "after_attn_residual"), after_attn_prefix);
    add_dump(capture, source_names, block_source_name(block, "norm2"), norm2_prefix);
    add_dump(capture,
             source_names,
             block_source_name(block, flat_mlp ? "mlp_fc1_flat" : "mlp_fc1"),
             mlp_fc1_prefix);
    add_dump(capture,
             source_names,
             block_source_name(block, flat_mlp ? "mlp_gelu_flat" : "mlp_gelu"),
             mlp_gelu_prefix);
    add_dump(capture, source_names, block_source_name(block, "mlp_fc2"), mlp_fc2_prefix);
    add_dump(capture, source_names, block_source_name(block, "out"), out_prefix);

    add_stage(capture, 0, "norm1", input_prefix);
    if (window_part_prefix.has_value()) {
        add_stage(capture, 1, "window_part", norm1_prefix);
    }
    add_stage(capture, 2, "qkv_proj", is_global ? norm1_prefix : *window_part_prefix);
    add_stage(capture, 3, "qkv_layout", qkv_prefix);
    add_stage(capture, 4, "qkv_rope", qkv_prefix);
    add_stage(capture, 5, "attn_core", qkv_prefix);
    add_stage(capture, 6, "attn_proj", attn_core_prefix);
    if (window_unpart_prefix.has_value()) {
        add_stage(capture, 7, "window_unpart", attn_proj_prefix);
    }
    add_stage(capture, 8, "norm2", after_attn_prefix);
    add_stage(capture, 9, "mlp_fc1", norm2_prefix);
    add_stage(capture, 10, "mlp_gelu", mlp_fc1_prefix);
    add_stage(capture, 11, "mlp_fc2", mlp_gelu_prefix);
    add_stage(capture, 12, "mlp", norm2_prefix);
    add_stage(capture, 13, "block", input_prefix);
    add_stage(capture, 14, "mlp_fc1_gelu", norm2_prefix);

    return capture;
}

[[nodiscard]] std::string manifest_json(std::string_view model_path,
                                        std::string_view backend,
                                        const std::filesystem::path& image_path,
                                        const std::filesystem::path& out_dir,
                                        const std::vector<BlockCapture>& captures) {
    std::string out = std::format(
        "{{\n  \"model\":\"{}\",\n  \"backend\":\"{}\",\n  \"image\":\"{}\",\n  "
        "\"out_dir\":\"{}\",\n  \"blocks\":[\n",
        json_escape(model_path),
        json_escape(backend),
        json_escape(image_path.string()),
        json_escape(out_dir.string()));
    for (size_t i = 0; i < captures.size(); ++i) {
        const auto& capture = captures[i];
        if (i != 0) {
            out += ",\n";
        }
        out += std::format(
            "    {{\"block\":{},\"is_global\":{},\"dumps\":[", capture.block, capture.is_global);
        for (size_t j = 0; j < capture.dumps.size(); ++j) {
            const auto& dump = capture.dumps[j];
            if (j != 0) {
                out += ",";
            }
            out += std::format("{{\"source\":\"{}\",\"prefix\":\"{}\"}}",
                               json_escape(dump.source),
                               json_escape(dump.prefix.string()));
        }
        out += "],\"stage_inputs\":[";
        for (size_t j = 0; j < capture.stage_inputs.size(); ++j) {
            const auto& stage = capture.stage_inputs[j];
            if (j != 0) {
                out += ",";
            }
            out += std::format("{{\"stage\":{},\"stage_name\":\"{}\",\"input_prefix\":\"{}\"}}",
                               stage.stage,
                               json_escape(stage.stage_name),
                               json_escape(stage.input_prefix.string()));
        }
        out += "]}";
    }
    out += "\n  ]\n}\n";
    return out;
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
        if (model_type != SAM3_MODEL_SAM3 && model_type != SAM3_MODEL_SAM3_1 &&
            model_type != SAM3_MODEL_SAM3_VISUAL) {
            std::cerr << "expected SAM3/SAM3 visual/SAM3.1 model\n";
            return 1;
        }

        auto state = sam3_create_state(*model, params);
        if (!state) {
            std::cerr << "failed to create state\n";
            return 1;
        }

        const sam3_image image = sam3_load_image(args.image_path.string());
        if (image.data.empty()) {
            std::cerr << std::format("failed to load image {}\n", args.image_path.string());
            return 1;
        }

        std::vector<int> blocks = args.blocks;
        std::ranges::sort(blocks);
        blocks.erase(std::ranges::unique(blocks).begin(), blocks.end());

        const int depth = sam3_test_vit_depth(*model);
        std::set<std::string> source_names;
        std::vector<BlockCapture> captures;
        captures.reserve(blocks.size());
        for (const int block : blocks) {
            if (block < 0 || block >= depth) {
                throw std::invalid_argument(
                    std::format("block {} out of range 0..{}", block, depth - 1));
            }
            captures.push_back(make_block_capture(args.out_dir,
                                                  block,
                                                  sam3_test_vit_block_is_global(*model, block),
                                                  sam3_test_vit_block_uses_flat_mlp(*model, block),
                                                  source_names));
        }

        std::vector<std::string> outputs(source_names.begin(), source_names.end());
        if (!sam3_encode_vit_from_image_selective(*state, *model, image, outputs)) {
            std::cerr << "selective ViT capture failed\n";
            return 1;
        }

        std::filesystem::create_directories(args.out_dir);
        for (const auto& capture : captures) {
            for (const auto& dump : capture.dumps) {
                if (!sam3_dump_state_tensor(*state, dump.source, dump.prefix.string())) {
                    std::cerr << std::format("failed to dump {}\n", dump.source);
                    return 1;
                }
            }
        }

        const auto manifest = manifest_json(
            args.model_path, sam3_backend_name(*model), args.image_path, args.out_dir, captures);
        const auto manifest_path = args.out_dir / "manifest.json";
        std::ofstream manifest_out(manifest_path);
        if (!manifest_out) {
            throw std::runtime_error(std::format("failed to open {}", manifest_path.string()));
        }
        manifest_out << manifest;
        std::cout << manifest;
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam3_vit_stage_capture failed: {}\n", ex.what());
        return 1;
    }
    return 0;
}
