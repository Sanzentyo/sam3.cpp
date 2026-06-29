#include "sam3.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::string model_path = "models/sam3.1/sam3.1_multiplex-f16.ggml";
    int threads = 4;
    bool use_gpu = true;
    int width = 320;
    int height = 240;
    int frame1_offset = 3;
    int num_frames = 2;
    int recondition_every = 16;
    std::string mask_case = "center";
    std::string output_dir;
    int warmup_runs = 0;
};

[[nodiscard]] int parse_positive_int(std::string_view text, std::string_view name) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value <= 0) {
        throw std::invalid_argument(std::format("invalid {}", name));
    }
    return value;
}

[[nodiscard]] int parse_non_negative_int(std::string_view text, std::string_view name) {
    int value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value < 0) {
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
        } else if (arg == "--threads") {
            args.threads = parse_positive_int(value(arg), arg);
        } else if (arg == "--width") {
            args.width = parse_positive_int(value(arg), arg);
        } else if (arg == "--height") {
            args.height = parse_positive_int(value(arg), arg);
        } else if (arg == "--frame1-offset") {
            args.frame1_offset = parse_non_negative_int(value(arg), arg);
        } else if (arg == "--num-frames") {
            args.num_frames = parse_positive_int(value(arg), arg);
        } else if (arg == "--recondition-every") {
            args.recondition_every = parse_positive_int(value(arg), arg);
        } else if (arg == "--warmup-runs") {
            args.warmup_runs = parse_non_negative_int(value(arg), arg);
        } else if (arg == "--mask-case") {
            args.mask_case = value(arg);
        } else if (arg == "--out") {
            args.output_dir = value(arg);
        } else if (arg == "--gpu") {
            args.use_gpu = true;
        } else if (arg == "--cpu") {
            args.use_gpu = false;
        } else {
            throw std::invalid_argument(std::format("unknown argument {}", arg));
        }
    }
    return args;
}

[[nodiscard]] sam3_image make_frame(int width, int height, int offset) {
    sam3_image image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t p =
                (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
            image.data[p + 0] = static_cast<uint8_t>((x + offset) % 256);
            image.data[p + 1] = static_cast<uint8_t>((y * 2 + offset) % 256);
            image.data[p + 2] = static_cast<uint8_t>((x + y + offset) % 256);
        }
    }
    return image;
}

struct Rect {
    int x0;
    int y0;
    int x1;
    int y1;
};

[[nodiscard]] int scale_dim(int size, int per_mille) {
    return std::clamp(size * per_mille / 1000, 0, size);
}

[[nodiscard]] Rect mask_case_rect(int width, int height, std::string_view mask_case) {
    if (mask_case == "center") {
        return {scale_dim(width, 250),
                scale_dim(height, 250),
                scale_dim(width, 750),
                scale_dim(height, 750)};
    }
    if (mask_case == "small-center") {
        return {scale_dim(width, 375),
                scale_dim(height, 375),
                scale_dim(width, 625),
                scale_dim(height, 625)};
    }
    if (mask_case == "left-wide") {
        return {scale_dim(width, 100),
                scale_dim(height, 200),
                scale_dim(width, 550),
                scale_dim(height, 800)};
    }
    if (mask_case == "bottom-band") {
        return {scale_dim(width, 200),
                scale_dim(height, 550),
                scale_dim(width, 800),
                scale_dim(height, 900)};
    }
    throw std::invalid_argument(std::format("unknown mask case {}", mask_case));
}

[[nodiscard]] sam3_detection make_rect_detection(int width,
                                                 int height,
                                                 std::string_view mask_case) {
    sam3_detection detection;
    detection.score = 1.0f;
    detection.iou_score = 1.0f;
    detection.obj_score_logit = 10.0f;
    detection.mask.width = width;
    detection.mask.height = height;
    detection.mask.iou_score = 1.0f;
    detection.mask.obj_score = 1.0f;
    detection.mask.data.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

    const auto rect = mask_case_rect(width, height, mask_case);
    detection.box = {
        .x0 = static_cast<float>(rect.x0),
        .y0 = static_cast<float>(rect.y0),
        .x1 = static_cast<float>(rect.x1),
        .y1 = static_cast<float>(rect.y1),
    };
    for (int y = rect.y0; y < rect.y1; ++y) {
        for (int x = rect.x0; x < rect.x1; ++x) {
            detection.mask.data[static_cast<size_t>(y) * static_cast<size_t>(width) +
                                static_cast<size_t>(x)] = 255;
        }
    }
    return detection;
}

void write_pgm(const sam3_mask& mask, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error(std::format("failed to open {}", path.string()));
    }
    out << "P5\n" << mask.width << " " << mask.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(mask.data.data()),
              static_cast<std::streamsize>(mask.data.size()));
}

[[nodiscard]] size_t count_foreground_pixels(const sam3_mask& mask) {
    return static_cast<size_t>(std::count_if(
        mask.data.begin(), mask.data.end(), [](uint8_t value) { return value > 127; }));
}

struct RunResult {
    int instance_id = -1;
    sam3_detection output;
    double model_load_ms = 0.0;
    double state_create_ms = 0.0;
    double tracker_create_ms = 0.0;
    double frame0_construct_ms = 0.0;
    double detection_construct_ms = 0.0;
    double frame1_construct_ms = 0.0;
    double frame_construct_total_ms = 0.0;
    double encode_frame0_ms = 0.0;
    double add_detection_ms = 0.0;
    double frame0_ms = 0.0;
    double encode_frame1_ms = 0.0;
    double tail_image_encode_total_ms = 0.0;
    double tail_image_encode_avg_ms = 0.0;
    double propagate_encoded_ms = 0.0;
    double propagate_ms = 0.0;
    double propagate_encoded_total_ms = 0.0;
    double propagate_encoded_avg_ms = 0.0;
    double track_step_total_ms = 0.0;
    double track_step_avg_ms = 0.0;
    double run_required_e2e_ms = 0.0;
    double one_shot_required_e2e_ms = 0.0;
    double output_write_ms = 0.0;
    int propagated_frames = 0;
    sam3_encode_timing frame0_encode_timing;
    sam3_encode_timing frame1_encode_timing;
    sam3_encode_timing tail_encode_timing_total;
    sam3_tracker_add_timing add_detection_timing;
    sam3_propagate_timing propagate_timing;

    [[nodiscard]] double required_session_setup_ms() const {
        return state_create_ms + tracker_create_ms;
    }

    [[nodiscard]] double required_input_prepare_ms() const {
        return frame_construct_total_ms + detection_construct_ms;
    }

    [[nodiscard]] double required_model_execute_ms() const {
        return frame0_ms + track_step_total_ms;
    }

    [[nodiscard]] double required_image_encode_ms() const {
        return encode_frame0_ms + tail_image_encode_total_ms;
    }

    [[nodiscard]] double required_mask_init_ms() const { return add_detection_ms; }

    [[nodiscard]] double required_propagate_encoded_ms() const {
        return propagate_encoded_total_ms;
    }

    [[nodiscard]] double required_model_accounted_ms() const {
        return required_image_encode_ms() + required_mask_init_ms() +
               required_propagate_encoded_ms();
    }

    [[nodiscard]] double required_model_remainder_ms() const {
        return required_model_execute_ms() - required_model_accounted_ms();
    }

    [[nodiscard]] double required_accounted_ms() const {
        return required_session_setup_ms() + required_input_prepare_ms() +
               required_model_execute_ms();
    }

    [[nodiscard]] double required_remainder_ms() const {
        return run_required_e2e_ms - required_accounted_ms();
    }
};

void add_encode_timing(sam3_encode_timing& total, const sam3_encode_timing& timing) {
    total.total_ms += timing.total_ms;
    total.preprocess_ms += timing.preprocess_ms;
    total.graph_build_ms += timing.graph_build_ms;
    total.graph_alloc_ms += timing.graph_alloc_ms;
    total.input_upload_ms += timing.input_upload_ms;
    total.graph_compute_ms += timing.graph_compute_ms;
    total.state_update_ms += timing.state_update_ms;
    total.pe_build_ms += timing.pe_build_ms;
    total.used_cuda_preprocess = total.used_cuda_preprocess || timing.used_cuda_preprocess;
    total.include_detector_neck = total.include_detector_neck || timing.include_detector_neck;
}

void write_encode_timing(std::ostream& out,
                         std::string_view key,
                         const sam3_encode_timing& timing,
                         bool trailing_comma) {
    out << std::format(
        "  \"{}\": {{\n"
        "    \"total_ms\": {:.6f},\n"
        "    \"preprocess_ms\": {:.6f},\n"
        "    \"graph_build_ms\": {:.6f},\n"
        "    \"graph_alloc_ms\": {:.6f},\n"
        "    \"input_upload_ms\": {:.6f},\n"
        "    \"graph_compute_ms\": {:.6f},\n"
        "    \"state_update_ms\": {:.6f},\n"
        "    \"pe_build_ms\": {:.6f},\n"
        "    \"used_cuda_preprocess\": {},\n"
        "    \"include_detector_neck\": {}\n"
        "  }}{}\n",
        key,
        timing.total_ms,
        timing.preprocess_ms,
        timing.graph_build_ms,
        timing.graph_alloc_ms,
        timing.input_upload_ms,
        timing.graph_compute_ms,
        timing.state_update_ms,
        timing.pe_build_ms,
        timing.used_cuda_preprocess,
        timing.include_detector_neck,
        trailing_comma ? "," : "");
}

void write_float_array(std::ostream& out,
                       std::string_view key,
                       const std::vector<float>& values,
                       bool trailing_comma) {
    out << std::format("  \"{}\": [", key);
    for (size_t i = 0; i < values.size(); ++i) {
        out << std::format("{}{:.9f}", i == 0 ? "" : ", ", values[i]);
    }
    out << std::format("]{}\n", trailing_comma ? "," : "");
}

void write_add_detection_timing(std::ostream& out,
                                const sam3_tracker_add_timing& timing,
                                bool trailing_comma) {
    out << std::format(
        "  \"add_detection_timing\": {{\n"
        "    \"mask_prepare_ms\": {:.6f},\n"
        "    \"memory_encode_ms\": {:.6f},\n"
        "    \"obj_ptr_ms\": {:.6f},\n"
        "    \"store_ms\": {:.6f}\n"
        "  }}{}\n",
        timing.mask_prepare_ms,
        timing.memory_encode_ms,
        timing.obj_ptr_ms,
        timing.store_ms,
        trailing_comma ? "," : "");
}

void write_propagate_timing(std::ostream& out,
                            const sam3_propagate_timing& timing,
                            bool trailing_comma) {
    out << std::format(
        "  \"propagate_timing\": {{\n"
        "    \"total_ms\": {:.6f},\n"
        "    \"prepare_caches_ms\": {:.6f},\n"
        "    \"memory_slot_read_ms\": {:.6f},\n"
        "    \"obj_ptr_read_ms\": {:.6f},\n"
        "    \"prompt_build_ms\": {:.6f},\n"
        "    \"memory_prepare_ms\": {:.6f},\n"
        "    \"ptr_prepare_ms\": {:.6f},\n"
        "    \"rope_cache_ms\": {:.6f},\n"
        "    \"rope_k_build_ms\": {:.6f},\n"
        "    \"graph_build_ms\": {:.6f},\n"
        "    \"graph_alloc_ms\": {:.6f},\n"
        "    \"input_upload_ms\": {:.6f},\n"
        "    \"input_prompt_upload_ms\": {:.6f},\n"
        "    \"input_rope_upload_ms\": {:.6f},\n"
        "    \"input_memory_upload_ms\": {:.6f},\n"
        "    \"input_feature_upload_ms\": {:.6f},\n"
        "    \"input_constant_upload_ms\": {:.6f},\n"
        "    \"input_sparse_upload_ms\": {:.6f},\n"
        "    \"graph_compute_ms\": {:.6f},\n"
        "    \"output_read_ms\": {:.6f},\n"
        "    \"graph_cache_build_ms\": {:.6f},\n"
        "    \"graph_cache_input_upload_ms\": {:.6f},\n"
        "    \"graph_cache_compute_ms\": {:.6f},\n"
        "    \"graph_cache_output_read_ms\": {:.6f},\n"
        "    \"bbox_from_logits_ms\": {:.6f},\n"
        "    \"fullmask_active_resize_bbox_ms\": {:.6f},\n"
        "    \"fullmask_pending_resize_bbox_ms\": {:.6f},\n"
        "    \"memory_update_ms\": {:.6f},\n"
        "    \"tracker_update_ms\": {:.6f},\n"
        "    \"result_build_ms\": {:.6f},\n"
        "    \"single_calls\": {},\n"
        "    \"active_masklets\": {},\n"
        "    \"pending_masklets\": {},\n"
        "    \"used_sam31_graph_cache\": {},\n"
        "    \"aliased_feature_inputs\": {},\n"
        "    \"aliased_constant_inputs\": {},\n"
        "    \"aliased_feature_input_count\": {},\n"
        "    \"aliased_constant_input_count\": {},\n"
        "    \"uploaded_feature_input_count\": {},\n"
        "    \"uploaded_constant_input_count\": {}\n"
        "  }}{}\n",
        timing.total_ms,
        timing.prepare_caches_ms,
        timing.memory_slot_read_ms,
        timing.obj_ptr_read_ms,
        timing.prompt_build_ms,
        timing.memory_prepare_ms,
        timing.ptr_prepare_ms,
        timing.rope_cache_ms,
        timing.rope_k_build_ms,
        timing.graph_build_ms,
        timing.graph_alloc_ms,
        timing.input_upload_ms,
        timing.input_prompt_upload_ms,
        timing.input_rope_upload_ms,
        timing.input_memory_upload_ms,
        timing.input_feature_upload_ms,
        timing.input_constant_upload_ms,
        timing.input_sparse_upload_ms,
        timing.graph_compute_ms,
        timing.output_read_ms,
        timing.graph_cache_build_ms,
        timing.graph_cache_input_upload_ms,
        timing.graph_cache_compute_ms,
        timing.graph_cache_output_read_ms,
        timing.bbox_from_logits_ms,
        timing.fullmask_active_resize_bbox_ms,
        timing.fullmask_pending_resize_bbox_ms,
        timing.memory_update_ms,
        timing.tracker_update_ms,
        timing.result_build_ms,
        timing.single_calls,
        timing.active_masklets,
        timing.pending_masklets,
        timing.used_sam31_graph_cache,
        timing.aliased_feature_inputs,
        timing.aliased_constant_inputs,
        timing.aliased_feature_input_count,
        timing.aliased_constant_input_count,
        timing.uploaded_feature_input_count,
        timing.uploaded_constant_input_count,
        trailing_comma ? "," : "");
}

void write_summary(const std::filesystem::path& path, const Args& args, const RunResult& run) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error(std::format("failed to open {}", path.string()));
    }
    f << std::format(
        "{{\n"
        "  \"status\": \"ok\",\n"
        "  \"mask_case\": \"{}\",\n"
        "  \"frame1_offset\": {},\n"
        "  \"num_frames\": {},\n"
        "  \"recondition_every\": {},\n"
        "  \"propagated_frames\": {},\n"
        "  \"warmup_runs\": {},\n"
        "  \"instance_id\": {},\n"
        "  \"detections\": 1,\n"
        "  \"mask_width\": {},\n"
        "  \"mask_height\": {},\n"
        "  \"score\": {:.9f},\n"
        "  \"obj_score_logit\": {:.9f},\n"
        "  \"model_load_ms\": {:.6f},\n"
        "  \"state_create_ms\": {:.6f},\n"
        "  \"tracker_create_ms\": {:.6f},\n"
        "  \"frame0_construct_ms\": {:.6f},\n"
        "  \"detection_construct_ms\": {:.6f},\n"
        "  \"frame1_construct_ms\": {:.6f},\n"
        "  \"frame_construct_total_ms\": {:.6f},\n"
        "  \"encode_frame0_ms\": {:.6f},\n"
        "  \"add_detection_ms\": {:.6f},\n"
        "  \"frame0_ms\": {:.6f},\n"
        "  \"encode_frame1_ms\": {:.6f},\n"
        "  \"tail_image_encode_total_ms\": {:.6f},\n"
        "  \"tail_image_encode_avg_ms\": {:.6f},\n"
        "  \"propagate_encoded_ms\": {:.6f},\n"
        "  \"propagate_ms\": {:.6f},\n"
        "  \"propagate_encoded_total_ms\": {:.6f},\n"
        "  \"propagate_encoded_avg_ms\": {:.6f},\n"
        "  \"track_step_total_ms\": {:.6f},\n"
        "  \"track_step_avg_ms\": {:.6f},\n"
        "  \"required_session_setup_ms\": {:.6f},\n"
        "  \"required_input_prepare_ms\": {:.6f},\n"
        "  \"required_model_execute_ms\": {:.6f},\n"
        "  \"required_image_encode_ms\": {:.6f},\n"
        "  \"required_mask_init_ms\": {:.6f},\n"
        "  \"required_propagate_encoded_ms\": {:.6f},\n"
        "  \"required_model_accounted_ms\": {:.6f},\n"
        "  \"required_model_remainder_ms\": {:.6f},\n"
        "  \"required_accounted_ms\": {:.6f},\n"
        "  \"required_remainder_ms\": {:.6f},\n"
        "  \"run_required_e2e_ms\": {:.6f},\n"
        "  \"one_shot_required_e2e_ms\": {:.6f},\n"
        "  \"mask_foreground_pixels\": {},\n"
        "  \"selected_mask_index\": {},\n"
        "  \"output_write_ms\": {:.6f},\n",
        args.mask_case,
        args.frame1_offset,
        args.num_frames,
        args.recondition_every,
        run.propagated_frames,
        args.warmup_runs,
        run.instance_id,
        run.output.mask.width,
        run.output.mask.height,
        run.output.score,
        run.output.obj_score_logit,
        run.model_load_ms,
        run.state_create_ms,
        run.tracker_create_ms,
        run.frame0_construct_ms,
        run.detection_construct_ms,
        run.frame1_construct_ms,
        run.frame_construct_total_ms,
        run.encode_frame0_ms,
        run.add_detection_ms,
        run.frame0_ms,
        run.encode_frame1_ms,
        run.tail_image_encode_total_ms,
        run.tail_image_encode_avg_ms,
        run.propagate_encoded_ms,
        run.propagate_ms,
        run.propagate_encoded_total_ms,
        run.propagate_encoded_avg_ms,
        run.track_step_total_ms,
        run.track_step_avg_ms,
        run.required_session_setup_ms(),
        run.required_input_prepare_ms(),
        run.required_model_execute_ms(),
        run.required_image_encode_ms(),
        run.required_mask_init_ms(),
        run.required_propagate_encoded_ms(),
        run.required_model_accounted_ms(),
        run.required_model_remainder_ms(),
        run.required_accounted_ms(),
        run.required_remainder_ms(),
        run.run_required_e2e_ms,
        run.one_shot_required_e2e_ms,
        count_foreground_pixels(run.output.mask),
        run.output.selected_mask_index,
        run.output_write_ms);
    write_float_array(f, "decoder_iou_scores", run.output.decoder_iou_scores, true);
    write_encode_timing(f, "frame0_encode_timing", run.frame0_encode_timing, true);
    write_encode_timing(f, "frame1_encode_timing", run.frame1_encode_timing, true);
    write_encode_timing(f, "tail_encode_timing_total", run.tail_encode_timing_total, true);
    write_add_detection_timing(f, run.add_detection_timing, true);
    write_propagate_timing(f, run.propagate_timing, false);
    f << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        if (args.num_frames < 2) {
            throw std::invalid_argument("--num-frames must be >= 2");
        }
        sam3_params params;
        params.model_path = args.model_path;
        params.n_threads = args.threads;
        params.use_gpu = args.use_gpu;
        params.require_gpu = args.use_gpu;

        const auto t_model_load0 = std::chrono::steady_clock::now();
        auto model = sam3_load_model(params);
        const auto t_model_load1 = std::chrono::steady_clock::now();
        const double model_load_ms =
            std::chrono::duration<double, std::milli>(t_model_load1 - t_model_load0).count();
        if (!model) {
            std::cerr << "failed to load SAM3.1 model\n";
            return 1;
        }
        if (sam3_get_model_type(*model) != SAM3_MODEL_SAM3_1) {
            std::cerr << "expected SAM3.1 model\n";
            return 1;
        }

        auto run_once = [&](bool write_outputs) -> RunResult {
            const auto t_run0 = std::chrono::steady_clock::now();
            auto state = sam3_create_state(*model, params);
            if (!state) {
                throw std::runtime_error("failed to create state");
            }
            const auto t_state = std::chrono::steady_clock::now();
            sam3_visual_track_params tracker_params;
            tracker_params.max_keep_alive = 4;
            tracker_params.recondition_every = args.recondition_every;
            auto tracker = sam3_create_visual_tracker(*model, tracker_params);
            if (!tracker) {
                throw std::runtime_error("failed to create tracker");
            }
            const auto t_tracker = std::chrono::steady_clock::now();

            const auto t_frame0_make0 = std::chrono::steady_clock::now();
            auto frame0 = make_frame(args.width, args.height, 0);
            const auto t_frame0_make1 = std::chrono::steady_clock::now();
            const auto t0 = std::chrono::steady_clock::now();
            if (!sam3_encode_image(*state, *model, frame0)) {
                throw std::runtime_error("frame0 encode failed");
            }
            const auto t_encode = std::chrono::steady_clock::now();
            const auto frame0_encode_timing = sam3_last_encode_timing();

            const auto t_detection_make0 = std::chrono::steady_clock::now();
            const auto detection = make_rect_detection(args.width, args.height, args.mask_case);
            const auto t_detection_make1 = std::chrono::steady_clock::now();
            const int instance_id = sam3_tracker_add_detection(*tracker, *state, *model, detection);
            if (instance_id < 0) {
                throw std::runtime_error("mask initialization failed");
            }
            const auto t_add = std::chrono::steady_clock::now();
            const auto add_detection_timing = sam3_tracker_last_add_timing();

            RunResult run;
            run.instance_id = instance_id;
            run.model_load_ms = model_load_ms;
            run.state_create_ms =
                std::chrono::duration<double, std::milli>(t_state - t_run0).count();
            run.tracker_create_ms =
                std::chrono::duration<double, std::milli>(t_tracker - t_state).count();
            run.frame0_construct_ms =
                std::chrono::duration<double, std::milli>(t_frame0_make1 - t_frame0_make0).count();
            run.detection_construct_ms =
                std::chrono::duration<double, std::milli>(t_detection_make1 - t_detection_make0)
                    .count();
            run.encode_frame0_ms = std::chrono::duration<double, std::milli>(t_encode - t0).count();
            run.add_detection_ms =
                std::chrono::duration<double, std::milli>(t_add - t_encode).count();
            run.frame0_ms = std::chrono::duration<double, std::milli>(t_add - t0).count();
            run.frame0_encode_timing = frame0_encode_timing;
            run.add_detection_timing = add_detection_timing;

            for (int frame_index = 1; frame_index < args.num_frames; ++frame_index) {
                const auto t_frame_make0 = std::chrono::steady_clock::now();
                const auto frame =
                    make_frame(args.width, args.height, args.frame1_offset * frame_index);
                const auto t_frame_make1 = std::chrono::steady_clock::now();
                const auto t_frame = std::chrono::steady_clock::now();
                if (!sam3_encode_image_for_tracking(*state, *model, frame)) {
                    throw std::runtime_error(
                        std::format("frame{} tracking encode failed", frame_index));
                }
                const auto t_frame_encoded = std::chrono::steady_clock::now();
                const auto frame_encode_timing = sam3_last_encode_timing();
                const auto result = sam3_propagate_encoded_frame(*tracker, *state, *model);
                const auto t_after_prop = std::chrono::steady_clock::now();
                const auto propagate_timing = sam3_last_propagate_timing();
                if (result.detections.empty() || result.detections.front().mask.data.empty()) {
                    throw std::runtime_error(
                        std::format("frame{} propagation produced no mask", frame_index));
                }

                const double frame_construct_ms =
                    std::chrono::duration<double, std::milli>(t_frame_make1 - t_frame_make0)
                        .count();
                const double encode_ms =
                    std::chrono::duration<double, std::milli>(t_frame_encoded - t_frame).count();
                const double propagate_encoded_ms =
                    std::chrono::duration<double, std::milli>(t_after_prop - t_frame_encoded)
                        .count();
                const double track_step_ms =
                    std::chrono::duration<double, std::milli>(t_after_prop - t_frame).count();
                run.frame_construct_total_ms += frame_construct_ms;
                if (frame_index == 1) {
                    run.output = result.detections.front();
                    run.frame1_construct_ms = frame_construct_ms;
                    run.encode_frame1_ms = encode_ms;
                    run.propagate_encoded_ms = propagate_encoded_ms;
                    run.propagate_ms =
                        std::chrono::duration<double, std::milli>(t_after_prop - t_add).count();
                    run.frame1_encode_timing = frame_encode_timing;
                    run.propagate_timing = propagate_timing;
                }
                run.propagate_encoded_total_ms += propagate_encoded_ms;
                run.tail_image_encode_total_ms += encode_ms;
                add_encode_timing(run.tail_encode_timing_total, frame_encode_timing);
                run.track_step_total_ms += track_step_ms;
                ++run.propagated_frames;
            }
            run.frame_construct_total_ms += run.frame0_construct_ms;
            run.propagate_encoded_avg_ms =
                run.propagate_encoded_total_ms / std::max(run.propagated_frames, 1);
            run.tail_image_encode_avg_ms =
                run.tail_image_encode_total_ms / std::max(run.propagated_frames, 1);
            run.track_step_avg_ms = run.track_step_total_ms / std::max(run.propagated_frames, 1);
            const auto t_run1 = std::chrono::steady_clock::now();
            run.run_required_e2e_ms =
                std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
            run.one_shot_required_e2e_ms = model_load_ms + run.run_required_e2e_ms;

            if (write_outputs && !args.output_dir.empty()) {
                const std::filesystem::path output_dir{args.output_dir};
                std::filesystem::create_directories(output_dir);
                const auto t_output0 = std::chrono::steady_clock::now();
                (void) sam3_save_mask(detection.mask, (output_dir / "input_mask.png").string());
                (void) sam3_save_mask(run.output.mask, (output_dir / "frame1_mask.png").string());
                write_pgm(detection.mask, output_dir / "input_mask.pgm");
                write_pgm(run.output.mask, output_dir / "frame1_mask.pgm");
                const auto t_output1 = std::chrono::steady_clock::now();
                run.output_write_ms =
                    std::chrono::duration<double, std::milli>(t_output1 - t_output0).count();
                write_summary(output_dir / "summary.json", args, run);
            }
            return run;
        };

        for (int i = 0; i < args.warmup_runs; ++i) {
            (void) run_once(false);
        }

        const auto run = run_once(true);
        std::cout << std::format(
            "sam31_tracking_mask_init_smoke ok: instance={} detections={} mask={}x{} score={:.6f} "
            "obj_logit={:.6f} encode_frame0_ms={:.3f} add_detection_ms={:.3f} "
            "frame0_ms={:.3f} encode_frame1_ms={:.3f} propagate_encoded_ms={:.3f} "
            "propagate_ms={:.3f} propagated_frames={} propagate_encoded_avg_ms={:.3f} "
            "track_step_avg_ms={:.3f}\n",
            run.instance_id,
            1,
            run.output.mask.width,
            run.output.mask.height,
            run.output.score,
            run.output.obj_score_logit,
            run.encode_frame0_ms,
            run.add_detection_ms,
            run.frame0_ms,
            run.encode_frame1_ms,
            run.propagate_encoded_ms,
            run.propagate_ms,
            run.propagated_frames,
            run.propagate_encoded_avg_ms,
            run.track_step_avg_ms);
    } catch (const std::exception& ex) {
        std::cerr << std::format("sam31_tracking_mask_init_smoke failed: {}\n", ex.what());
        return 1;
    }
    return 0;
}
