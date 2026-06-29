/**
 * sam3_benchmark — exhaustive latency benchmark for all model variants.
 *
 * Tracks an object (point prompt) across N video frames on CPU and GPU,
 * then prints a formatted comparison table.
 *
 * On POSIX systems each model × backend run is isolated in a forked subprocess
 * so that a crash (e.g. unsupported Metal op) does not kill the entire benchmark.
 * On Windows, benchmarks run in-process (no crash isolation).
 *
 * Usage:
 *   sam3_benchmark [options]
 *
 * Options:
 *   --models-dir <path>   Directory with .ggml files  (default: models/)
 *   --video <path>        Video file                   (default: data/test_video.mp4)
 *   --frame-dir <path>    Directory of extracted frames, used instead of --video
 *   --point-x <f>         Click point X                (default: 315.0)
 *   --point-y <f>         Click point Y                (default: 250.0)
 *   --text-prompt <text>  SAM3 text prompt             (default: person)
 *   --n-frames <n>        Frames to track              (default: 10)
 *   --n-threads <n>       CPU threads                  (default: 4)
 *   --recondition-every <n> Memory refresh interval    (default: 16)
 *   --cpu-only            Skip GPU runs
 *   --gpu-only            Skip CPU runs
 *   --bbox-only           Track/output bbox rows without full-res masks
 *   --multimask           Use multimask output for the initial point prompt
 *   --initial-candidate-index <n> Force a point-prompt candidate for diagnostics
 *   --filter <substr>     Only run models whose filename contains <substr>
 *   --output-jsonl <path> Write first-run target bbox rows for quality checks
 *   --output-initial-candidates-jsonl <path> Write initial point-prompt candidates
 *   --output-frame-timing-jsonl <path> Write per-frame timing rows after measurement
 *   --output-mask-dir <path> Write first-run target masks as PNG files
 *   --output-logits-dir <path> Write first-run selected low-res logits as .bin/.shape
 *   --no-isolation        Run in-process for profilers that do not follow fork
 */

#include "sam3.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <ostream>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <sys/stat.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
#include <span>
#endif

// ── Wire format for child→parent result ─────────────────────────────────────

struct BenchWire {
    double t_load_ms;
    double t_frame0_ms;
    double t_frame0_encode_ms;
    double t_frame0_prompt_ms;
    double t_frame0_segment_ms;
    double t_frame0_tracker_add_ms;
    double t_frame0_tracker_mask_prepare_ms;
    double t_frame0_tracker_memory_encode_ms;
    double t_frame0_tracker_obj_ptr_ms;
    double t_frame0_tracker_store_ms;
    double t_frame0_tracker_unattributed_ms;
    double t_frame0_prompt_unattributed_ms;
    double t_track_avg_ms;
    double t_track_p50_ms;
    double t_track_p95_ms;
    double t_total_ms;
    double t_session_ms;
    double t_input_load_ms;
    double t_input_directory_scan_ms;
    double t_input_sort_ms;
    double t_input_frame_decode_ms;
    double t_input_frame_store_ms;
    double t_input_remainder_ms;
    double t_model_e2e_ms;
    double t_required_e2e_ms;
    double t_cold_required_e2e_ms;
    double t_input_plus_session_ms;
    double t_required_accounted_ms;
    double t_required_remainder_ms;
    double t_required_accounting_delta_ms;
    double t_model_accounting_delta_ms;
    double t_model_accounted_ms;
    double t_session_overhead_ms;
    double t_model_remainder_ms;
    double t_state_create_ms;
    double t_tracker_create_ms;
    double t_session_setup_ms;
    double t_artifact_ms;
    double t_frame0_artifact_ms;
    double t_frame0_model_ms;
    double t_frame0_prompt_model_ms;
    double t_track_all_ms;
    double t_track_all_model_ms;
    double t_tail_model_ms;
    double t_tail_encode_required_total_ms;
    double t_required_tail_state_create_ms;
    double t_required_tail_encode_ms;
    double t_required_tail_propagate_ms;
    double t_required_tail_unattributed_ms;
    double t_required_output_artifact_ms;
    double t_tail_inline_encode_ms;
    double t_tail_preencode_encode_ms;
    double t_tail_inline_propagate_ms;
    double t_tail_preencoded_propagate_ms;
    double t_frame0_encode_graph_compute_ms;
    double t_tail_inline_encode_graph_compute_ms;
    double t_tail_preencode_encode_graph_compute_ms;
    double t_tail_encode_graph_compute_ms;
    double t_tail_propagate_graph_compute_ms;
    double t_tail_propagate_cache_compute_ms;
    double t_tail_propagate_input_prompt_upload_ms;
    double t_tail_propagate_input_rope_upload_ms;
    double t_tail_propagate_input_memory_upload_ms;
    double t_tail_propagate_input_feature_upload_ms;
    double t_tail_propagate_input_constant_upload_ms;
    double t_tail_propagate_input_sparse_upload_ms;
    double t_required_core_compute_ms;
    double t_required_non_core_compute_ms;
    double t_required_core_compute_fraction;
    double t_tail_encode_graph_compute_fraction;
    double t_track_encode_avg_ms;
    double t_track_encode_total_ms;
    double t_track_propagate_avg_ms;
    double t_track_propagate_total_ms;
    double t_track_unattributed_avg_ms;
    double t_track_unattributed_total_ms;
    double t_track_artifact_avg_ms;
    double t_track_artifact_total_ms;
    double t_preencode_avg_ms;
    double t_preencode_total_ms;
    double t_preencode_state_create_avg_ms;
    double t_preencode_state_create_total_ms;
    sam3_encode_timing frame0_encode_timing;
    int n_track_frames;
    int n_track_all_frames;
    int n_track_split_frames;
    int n_preencoded_frames;
    int n_frame0_candidates;
    int n_frame0_added_instances;
    int n_input_frames;
    int input_from_frame_dir;
    int text_init_selected_only;
    long max_rss_kib;
    int n_detections;
    int ok;  // 1 = success, 0 = failure
    std::array<char, 32> backend;
    std::array<char, 128> error;
};

static_assert(std::is_trivially_copyable_v<BenchWire>);

// ── Display result ──────────────────────────────────────────────────────────

struct BenchResult {
    std::string model_name;
    std::string backend;
    int64_t file_size = 0;
    double t_load_ms = 0.0;
    double t_frame0_ms = 0.0;
    double t_frame0_encode_ms = 0.0;
    double t_frame0_prompt_ms = 0.0;
    double t_frame0_segment_ms = 0.0;
    double t_frame0_tracker_add_ms = 0.0;
    double t_frame0_tracker_mask_prepare_ms = 0.0;
    double t_frame0_tracker_memory_encode_ms = 0.0;
    double t_frame0_tracker_obj_ptr_ms = 0.0;
    double t_frame0_tracker_store_ms = 0.0;
    double t_frame0_tracker_unattributed_ms = 0.0;
    double t_frame0_prompt_unattributed_ms = 0.0;
    double t_track_avg_ms = 0.0;
    double t_track_p50_ms = 0.0;
    double t_track_p95_ms = 0.0;
    double t_total_ms = 0.0;
    double t_session_ms = 0.0;
    double t_input_load_ms = 0.0;
    double t_input_directory_scan_ms = 0.0;
    double t_input_sort_ms = 0.0;
    double t_input_frame_decode_ms = 0.0;
    double t_input_frame_store_ms = 0.0;
    double t_input_remainder_ms = 0.0;
    double t_model_e2e_ms = 0.0;
    double t_required_e2e_ms = 0.0;
    double t_cold_required_e2e_ms = 0.0;
    double t_input_plus_session_ms = 0.0;
    double t_required_accounted_ms = 0.0;
    double t_required_remainder_ms = 0.0;
    double t_required_accounting_delta_ms = 0.0;
    double t_model_accounting_delta_ms = 0.0;
    double t_model_accounted_ms = 0.0;
    double t_session_overhead_ms = 0.0;
    double t_model_remainder_ms = 0.0;
    double t_state_create_ms = 0.0;
    double t_tracker_create_ms = 0.0;
    double t_session_setup_ms = 0.0;
    double t_artifact_ms = 0.0;
    double t_frame0_artifact_ms = 0.0;
    double t_frame0_model_ms = 0.0;
    double t_frame0_prompt_model_ms = 0.0;
    double t_track_all_ms = 0.0;
    double t_track_all_model_ms = 0.0;
    double t_tail_model_ms = 0.0;
    double t_tail_encode_required_total_ms = 0.0;
    double t_required_tail_state_create_ms = 0.0;
    double t_required_tail_encode_ms = 0.0;
    double t_required_tail_propagate_ms = 0.0;
    double t_required_tail_unattributed_ms = 0.0;
    double t_required_output_artifact_ms = 0.0;
    double t_tail_inline_encode_ms = 0.0;
    double t_tail_preencode_encode_ms = 0.0;
    double t_tail_inline_propagate_ms = 0.0;
    double t_tail_preencoded_propagate_ms = 0.0;
    double t_frame0_encode_graph_compute_ms = 0.0;
    double t_tail_inline_encode_graph_compute_ms = 0.0;
    double t_tail_preencode_encode_graph_compute_ms = 0.0;
    double t_tail_encode_graph_compute_ms = 0.0;
    double t_tail_propagate_graph_compute_ms = 0.0;
    double t_tail_propagate_cache_compute_ms = 0.0;
    double t_tail_propagate_input_prompt_upload_ms = 0.0;
    double t_tail_propagate_input_rope_upload_ms = 0.0;
    double t_tail_propagate_input_memory_upload_ms = 0.0;
    double t_tail_propagate_input_feature_upload_ms = 0.0;
    double t_tail_propagate_input_constant_upload_ms = 0.0;
    double t_tail_propagate_input_sparse_upload_ms = 0.0;
    double t_required_core_compute_ms = 0.0;
    double t_required_non_core_compute_ms = 0.0;
    double t_required_core_compute_fraction = 0.0;
    double t_tail_encode_graph_compute_fraction = 0.0;
    double t_track_encode_avg_ms = 0.0;
    double t_track_encode_total_ms = 0.0;
    double t_track_propagate_avg_ms = 0.0;
    double t_track_propagate_total_ms = 0.0;
    double t_track_unattributed_avg_ms = 0.0;
    double t_track_unattributed_total_ms = 0.0;
    double t_track_artifact_avg_ms = 0.0;
    double t_track_artifact_total_ms = 0.0;
    double t_preencode_avg_ms = 0.0;
    double t_preencode_total_ms = 0.0;
    double t_preencode_state_create_avg_ms = 0.0;
    double t_preencode_state_create_total_ms = 0.0;
    int n_track_frames = 0;
    int n_track_all_frames = 0;
    int n_track_split_frames = 0;
    int n_preencoded_frames = 0;
    int n_frame0_candidates = 0;
    int n_frame0_added_instances = 0;
    int n_input_frames = 0;
    int input_from_frame_dir = 0;
    int text_init_selected_only = 0;
    long max_rss_kib = 0;
    int n_detections = 0;
    bool success = false;
    std::string error;
};

struct FrameTimingRow {
    int frame_index = 0;
    bool timed = false;
    bool split_timing = false;
    bool preencoded = false;
    int detections = 0;
    double total_ms = 0.0;
    double model_ms = 0.0;
    double runtime_wall_ms = 0.0;
    double visible_model_ms = 0.0;
    double preencode_required_model_ms = 0.0;
    double required_model_ms = 0.0;
    double required_wall_ms = 0.0;
    double inline_encode_ms = 0.0;
    double preencode_state_create_ms = 0.0;
    double preencode_ms = 0.0;
    double required_encode_ms = 0.0;
    double propagate_ms = 0.0;
    double artifact_ms = 0.0;
    double unattributed_ms = 0.0;
    sam3_encode_timing encode_timing;
    sam3_propagate_timing propagate_timing;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string format_size(int64_t bytes) {
    if (bytes >= (int64_t) 1024 * 1024 * 1024) {
        return std::format("{:.1f} GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
    if (bytes >= (int64_t) 1024 * 1024) {
        return std::format("{} MB", bytes / (1024 * 1024));
    }
    return std::format("{} KB", bytes / 1024);
}

static std::string format_time_short(double ms) {
    if (ms < 0)
        return "  -";
    return std::format("{:.1f}", ms);
}

static std::optional<size_t> parse_size_t(std::string_view value) {
    size_t parsed = 0;
    const auto* first = value.data();
    const auto* last = first + value.size();
    const auto [ptr, ec] = std::from_chars(first, last, parsed);
    if (ec != std::errc{} || ptr != last) {
        return std::nullopt;
    }
    return parsed;
}

#ifndef _WIN32
static long max_rss_to_kib(long ru_maxrss) {
#if defined(__APPLE__) && defined(__MACH__)
    return (ru_maxrss + 1023) / 1024;
#else
    return ru_maxrss;
#endif
}

class DirHandle {
public:
    DirHandle() = default;

    explicit DirHandle(DIR* dir) : dir_(dir) {}

    DirHandle(const DirHandle&) = delete;
    DirHandle& operator=(const DirHandle&) = delete;

    DirHandle(DirHandle&& other) noexcept : dir_(std::exchange(other.dir_, nullptr)) {}

    DirHandle& operator=(DirHandle&& other) noexcept {
        if (this != &other) {
            reset();
            dir_ = std::exchange(other.dir_, nullptr);
        }
        return *this;
    }

    ~DirHandle() { reset(); }

    [[nodiscard]] DIR* get() const noexcept { return dir_; }

    void reset(DIR* dir = nullptr) noexcept {
        if (dir_) {
            closedir(dir_);
        }
        dir_ = dir;
    }

    [[nodiscard]] static std::expected<DirHandle, std::string> open(std::string_view path) {
        std::string owned_path{path};
        DIR* dir = opendir(owned_path.c_str());
        if (!dir) {
            return std::unexpected("cannot open models directory");
        }
        return DirHandle{dir};
    }

private:
    DIR* dir_ = nullptr;
};

class ScopedFd {
public:
    ScopedFd() = default;

    explicit ScopedFd(int fd) : fd_(fd) {}

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    ScopedFd(ScopedFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

    ScopedFd& operator=(ScopedFd&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    ~ScopedFd() { reset(); }

    [[nodiscard]] int get() const noexcept { return fd_; }

    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

    void reset(int fd = -1) noexcept {
        if (fd_ >= 0) {
            close(fd_);
        }
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

struct Pipe {
    ScopedFd read;
    ScopedFd write;
};

[[nodiscard]] static std::expected<Pipe, std::string> make_pipe() {
    std::array<int, 2> fds = {-1, -1};
    if (pipe(fds.data()) != 0) {
        return std::unexpected("pipe() failed");
    }
    return Pipe{ScopedFd{fds[0]}, ScopedFd{fds[1]}};
}

static bool write_full(int fd, std::span<const std::byte> bytes) {
    const std::byte* p = bytes.data();
    size_t done = 0;
    while (done < bytes.size()) {
        ssize_t n = write(fd, p + done, bytes.size() - done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        done += (size_t) n;
    }
    return true;
}

static bool read_full(int fd, std::span<std::byte> bytes) {
    std::byte* p = bytes.data();
    size_t done = 0;
    while (done < bytes.size()) {
        ssize_t n = read(fd, p + done, bytes.size() - done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        done += (size_t) n;
    }
    return true;
}
#endif

static bool ends_with(std::string_view s, std::string_view suffix) {
    if (suffix.size() > s.size())
        return false;
    return s.substr(s.size() - suffix.size()) == suffix;
}

static std::string strip_extension(std::string_view filename) {
    auto pos = filename.rfind('.');
    return std::string{(pos != std::string_view::npos) ? filename.substr(0, pos) : filename};
}

static bool is_supported_frame_file(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::ranges::transform(
        ext, ext.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

struct BenchmarkInputTiming {
    double directory_scan_ms = 0.0;
    double sort_ms = 0.0;
    double frame_decode_ms = 0.0;
    double frame_store_ms = 0.0;
    int frames = 0;
    bool from_frame_dir = false;
};

struct BenchmarkInputFrames {
    std::vector<sam3_image> frames;
    BenchmarkInputTiming timing;
};

static double elapsed_ms(int64_t start_us) {
    return (ggml_time_us() - start_us) / 1000.0;
}

static std::expected<BenchmarkInputFrames, std::string> load_benchmark_frames(
    const std::string& video_path, const std::string& frame_dir, int n_frames) {
    BenchmarkInputFrames result;
    result.frames.reserve(static_cast<size_t>(n_frames));
    result.timing.from_frame_dir = !frame_dir.empty();

    if (!frame_dir.empty()) {
        std::vector<std::filesystem::path> paths;
        std::error_code ec;
        const int64_t scan_t0 = ggml_time_us();
        for (const auto& entry : std::filesystem::directory_iterator(frame_dir, ec)) {
            if (!entry.is_regular_file(ec) || !is_supported_frame_file(entry.path())) {
                continue;
            }
            paths.push_back(entry.path());
        }
        result.timing.directory_scan_ms = elapsed_ms(scan_t0);
        if (ec) {
            return std::unexpected(std::format("cannot read frame directory: {}", frame_dir));
        }
        const int64_t sort_t0 = ggml_time_us();
        std::ranges::sort(paths);
        result.timing.sort_ms = elapsed_ms(sort_t0);
        if (std::cmp_less(paths.size(), n_frames)) {
            return std::unexpected(std::format(
                "frame directory has {} usable frames, need {}", paths.size(), n_frames));
        }
        for (int f = 0; f < n_frames; ++f) {
            const auto& path = paths[static_cast<size_t>(f)];
            const int64_t decode_t0 = ggml_time_us();
            auto image = sam3_load_image(path.string());
            result.timing.frame_decode_ms += elapsed_ms(decode_t0);
            if (image.data.empty()) {
                return std::unexpected(std::format("load frame failed: {}", path.string()));
            }
            const int64_t store_t0 = ggml_time_us();
            result.frames.push_back(std::move(image));
            result.timing.frame_store_ms += elapsed_ms(store_t0);
            ++result.timing.frames;
        }
        return result;
    }

    for (int f = 0; f < n_frames; ++f) {
        const int64_t decode_t0 = ggml_time_us();
        auto image = sam3_decode_video_frame(video_path, f);
        result.timing.frame_decode_ms += elapsed_ms(decode_t0);
        if (image.data.empty()) {
            return std::unexpected("decode frame failed");
        }
        const int64_t store_t0 = ggml_time_us();
        result.frames.push_back(std::move(image));
        result.timing.frame_store_ms += elapsed_ms(store_t0);
        ++result.timing.frames;
    }
    return result;
}

static const char* requested_backend_label(bool use_gpu) {
    if (!use_gpu)
        return "CPU";
#ifdef GGML_USE_CUDA
    return "CUDA";
#elif defined(GGML_USE_METAL)
    return "Metal";
#else
    return "GPU";
#endif
}

static std::string json_escape(std::string_view s);

static std::string display_backend_name(std::string_view backend_name, bool requested_gpu) {
    if (!requested_gpu)
        return "CPU";
    if (backend_name.starts_with("CUDA"))
        return "CUDA";
    if (backend_name.starts_with("Metal"))
        return "Metal";
    return backend_name.empty() ? requested_backend_label(true) : std::string{backend_name};
}

static void apply_successful_wire(BenchResult& res, const BenchWire& wire, bool use_gpu) {
    res.t_load_ms = wire.t_load_ms;
    res.t_frame0_ms = wire.t_frame0_ms;
    res.t_frame0_encode_ms = wire.t_frame0_encode_ms;
    res.t_frame0_prompt_ms = wire.t_frame0_prompt_ms;
    res.t_frame0_segment_ms = wire.t_frame0_segment_ms;
    res.t_frame0_tracker_add_ms = wire.t_frame0_tracker_add_ms;
    res.t_frame0_tracker_mask_prepare_ms = wire.t_frame0_tracker_mask_prepare_ms;
    res.t_frame0_tracker_memory_encode_ms = wire.t_frame0_tracker_memory_encode_ms;
    res.t_frame0_tracker_obj_ptr_ms = wire.t_frame0_tracker_obj_ptr_ms;
    res.t_frame0_tracker_store_ms = wire.t_frame0_tracker_store_ms;
    res.t_frame0_tracker_unattributed_ms = wire.t_frame0_tracker_unattributed_ms;
    res.t_frame0_prompt_unattributed_ms = wire.t_frame0_prompt_unattributed_ms;
    res.t_track_avg_ms = wire.t_track_avg_ms;
    res.t_track_p50_ms = wire.t_track_p50_ms;
    res.t_track_p95_ms = wire.t_track_p95_ms;
    res.t_total_ms = wire.t_total_ms;
    res.t_session_ms = wire.t_session_ms;
    res.t_input_load_ms = wire.t_input_load_ms;
    res.t_input_directory_scan_ms = wire.t_input_directory_scan_ms;
    res.t_input_sort_ms = wire.t_input_sort_ms;
    res.t_input_frame_decode_ms = wire.t_input_frame_decode_ms;
    res.t_input_frame_store_ms = wire.t_input_frame_store_ms;
    res.t_input_remainder_ms = wire.t_input_remainder_ms;
    res.t_model_e2e_ms = wire.t_model_e2e_ms;
    res.t_required_e2e_ms = wire.t_required_e2e_ms;
    res.t_cold_required_e2e_ms = wire.t_cold_required_e2e_ms;
    res.t_input_plus_session_ms = wire.t_input_plus_session_ms;
    res.t_required_accounted_ms = wire.t_required_accounted_ms;
    res.t_required_remainder_ms = wire.t_required_remainder_ms;
    res.t_required_accounting_delta_ms = wire.t_required_accounting_delta_ms;
    res.t_model_accounting_delta_ms = wire.t_model_accounting_delta_ms;
    res.t_model_accounted_ms = wire.t_model_accounted_ms;
    res.t_session_overhead_ms = wire.t_session_overhead_ms;
    res.t_model_remainder_ms = wire.t_model_remainder_ms;
    res.t_state_create_ms = wire.t_state_create_ms;
    res.t_tracker_create_ms = wire.t_tracker_create_ms;
    res.t_session_setup_ms = wire.t_session_setup_ms;
    res.t_artifact_ms = wire.t_artifact_ms;
    res.t_frame0_artifact_ms = wire.t_frame0_artifact_ms;
    res.t_frame0_model_ms = wire.t_frame0_model_ms;
    res.t_frame0_prompt_model_ms = wire.t_frame0_prompt_model_ms;
    res.t_track_all_ms = wire.t_track_all_ms;
    res.t_track_all_model_ms = wire.t_track_all_model_ms;
    res.t_tail_model_ms = wire.t_tail_model_ms;
    res.t_tail_encode_required_total_ms = wire.t_tail_encode_required_total_ms;
    res.t_required_tail_state_create_ms = wire.t_required_tail_state_create_ms;
    res.t_required_tail_encode_ms = wire.t_required_tail_encode_ms;
    res.t_required_tail_propagate_ms = wire.t_required_tail_propagate_ms;
    res.t_required_tail_unattributed_ms = wire.t_required_tail_unattributed_ms;
    res.t_required_output_artifact_ms = wire.t_required_output_artifact_ms;
    res.t_tail_inline_encode_ms = wire.t_tail_inline_encode_ms;
    res.t_tail_preencode_encode_ms = wire.t_tail_preencode_encode_ms;
    res.t_tail_inline_propagate_ms = wire.t_tail_inline_propagate_ms;
    res.t_tail_preencoded_propagate_ms = wire.t_tail_preencoded_propagate_ms;
    res.t_frame0_encode_graph_compute_ms = wire.t_frame0_encode_graph_compute_ms;
    res.t_tail_inline_encode_graph_compute_ms = wire.t_tail_inline_encode_graph_compute_ms;
    res.t_tail_preencode_encode_graph_compute_ms = wire.t_tail_preencode_encode_graph_compute_ms;
    res.t_tail_encode_graph_compute_ms = wire.t_tail_encode_graph_compute_ms;
    res.t_tail_propagate_graph_compute_ms = wire.t_tail_propagate_graph_compute_ms;
    res.t_tail_propagate_cache_compute_ms = wire.t_tail_propagate_cache_compute_ms;
    res.t_tail_propagate_input_prompt_upload_ms = wire.t_tail_propagate_input_prompt_upload_ms;
    res.t_tail_propagate_input_rope_upload_ms = wire.t_tail_propagate_input_rope_upload_ms;
    res.t_tail_propagate_input_memory_upload_ms = wire.t_tail_propagate_input_memory_upload_ms;
    res.t_tail_propagate_input_feature_upload_ms = wire.t_tail_propagate_input_feature_upload_ms;
    res.t_tail_propagate_input_constant_upload_ms = wire.t_tail_propagate_input_constant_upload_ms;
    res.t_tail_propagate_input_sparse_upload_ms = wire.t_tail_propagate_input_sparse_upload_ms;
    res.t_required_core_compute_ms = wire.t_required_core_compute_ms;
    res.t_required_non_core_compute_ms = wire.t_required_non_core_compute_ms;
    res.t_required_core_compute_fraction = wire.t_required_core_compute_fraction;
    res.t_tail_encode_graph_compute_fraction = wire.t_tail_encode_graph_compute_fraction;
    res.t_track_encode_avg_ms = wire.t_track_encode_avg_ms;
    res.t_track_encode_total_ms = wire.t_track_encode_total_ms;
    res.t_track_propagate_avg_ms = wire.t_track_propagate_avg_ms;
    res.t_track_propagate_total_ms = wire.t_track_propagate_total_ms;
    res.t_track_unattributed_avg_ms = wire.t_track_unattributed_avg_ms;
    res.t_track_unattributed_total_ms = wire.t_track_unattributed_total_ms;
    res.t_track_artifact_avg_ms = wire.t_track_artifact_avg_ms;
    res.t_track_artifact_total_ms = wire.t_track_artifact_total_ms;
    res.t_preencode_avg_ms = wire.t_preencode_avg_ms;
    res.t_preencode_total_ms = wire.t_preencode_total_ms;
    res.t_preencode_state_create_avg_ms = wire.t_preencode_state_create_avg_ms;
    res.t_preencode_state_create_total_ms = wire.t_preencode_state_create_total_ms;
    res.n_track_frames = wire.n_track_frames;
    res.n_track_all_frames = wire.n_track_all_frames;
    res.n_track_split_frames = wire.n_track_split_frames;
    res.n_preencoded_frames = wire.n_preencoded_frames;
    res.n_frame0_candidates = wire.n_frame0_candidates;
    res.n_frame0_added_instances = wire.n_frame0_added_instances;
    res.n_input_frames = wire.n_input_frames;
    res.input_from_frame_dir = wire.input_from_frame_dir;
    res.text_init_selected_only = wire.text_init_selected_only;
    res.max_rss_kib = wire.max_rss_kib;
    res.n_detections = wire.n_detections;
    res.backend = display_backend_name(wire.backend.data(), use_gpu);
    res.success = true;
}

static void print_result_json(size_t index, const BenchResult& r) {
    const std::string model_json = json_escape(r.model_name);
    const std::string backend_json = json_escape(r.backend);
    const std::string_view input_source = r.input_from_frame_dir != 0 ? "frame_dir" : "video";
    if (!r.success) {
        const std::string error_json = json_escape(r.error);
        std::println(
            "SAM3_BENCH_RESULT_JSON "
            "{{\"index\":{},\"model\":\"{}\",\"size\":\"{}\",\"backend\":\"{}\","
            "\"ok\":false,\"error\":\"{}\"}}",
            index,
            model_json,
            format_size(r.file_size),
            backend_json,
            error_json);
        return;
    }

    std::println(
        "SAM3_BENCH_RESULT_JSON "
        "{{\"index\":{},\"model\":\"{}\",\"size\":\"{}\",\"backend\":\"{}\","
        "\"ok\":true,"
        "\"load_ms\":{:.6f},"
        "\"model_load_ms\":{:.6f},"
        "\"frame0_ms\":{:.6f},"
        "\"frame0_encode_ms\":{:.6f},"
        "\"frame0_prompt_ms\":{:.6f},"
        "\"frame0_segment_ms\":{:.6f},"
        "\"frame0_tracker_add_ms\":{:.6f},"
        "\"frame0_tracker_mask_prepare_ms\":{:.6f},"
        "\"frame0_tracker_memory_encode_ms\":{:.6f},"
        "\"frame0_tracker_obj_ptr_ms\":{:.6f},"
        "\"frame0_tracker_store_ms\":{:.6f},"
        "\"frame0_tracker_unattributed_ms\":{:.6f},"
        "\"frame0_prompt_unattributed_ms\":{:.6f},"
        "\"track_ms\":{:.6f},"
        "\"p50_ms\":{:.6f},"
        "\"p95_ms\":{:.6f},"
        "\"total_ms\":{:.6f},"
        "\"session_ms\":{:.6f},"
        "\"input_load_ms\":{:.6f},"
        "\"input_directory_scan_ms\":{:.6f},"
        "\"input_sort_ms\":{:.6f},"
        "\"input_frame_decode_ms\":{:.6f},"
        "\"input_frame_store_ms\":{:.6f},"
        "\"input_remainder_ms\":{:.6f},"
        "\"input_frames\":{},"
        "\"input_source\":\"{}\","
        "\"model_e2e_ms\":{:.6f},"
        "\"required_e2e_ms\":{:.6f},"
        "\"cold_required_e2e_ms\":{:.6f},"
        "\"input_plus_session_ms\":{:.6f},"
        "\"required_accounted_ms\":{:.6f},"
        "\"required_remainder_ms\":{:.6f},"
        "\"required_accounting_delta_ms\":{:.6f},"
        "\"model_accounting_delta_ms\":{:.6f},"
        "\"model_accounted_ms\":{:.6f},"
        "\"session_overhead_ms\":{:.6f},"
        "\"model_remainder_ms\":{:.6f},"
        "\"state_create_ms\":{:.6f},"
        "\"tracker_create_ms\":{:.6f},"
        "\"session_setup_ms\":{:.6f},"
        "\"artifact_ms\":{:.6f},"
        "\"frame0_artifact_ms\":{:.6f},"
        "\"frame0_model_ms\":{:.6f},"
        "\"frame0_prompt_model_ms\":{:.6f},"
        "\"track_all_ms\":{:.6f},"
        "\"track_all_model_ms\":{:.6f},"
        "\"tail_model_ms\":{:.6f},"
        "\"tail_runtime_wall_ms\":{:.6f},"
        "\"tail_visible_model_ms\":{:.6f},"
        "\"tail_preencode_required_model_ms\":{:.6f},"
        "\"tail_required_model_ms\":{:.6f},"
        "\"tail_encode_required_total_ms\":{:.6f},"
        "\"required_tail_state_create_ms\":{:.6f},"
        "\"required_tail_encode_ms\":{:.6f},"
        "\"required_tail_propagate_ms\":{:.6f},"
        "\"required_tail_unattributed_ms\":{:.6f},"
        "\"required_output_artifact_ms\":{:.6f},"
        "\"tail_inline_encode_ms\":{:.6f},"
        "\"tail_preencode_encode_ms\":{:.6f},"
        "\"tail_inline_propagate_ms\":{:.6f},"
        "\"tail_preencoded_propagate_ms\":{:.6f},"
        "\"frame0_encode_graph_compute_ms\":{:.6f},"
        "\"tail_inline_encode_graph_compute_ms\":{:.6f},"
        "\"tail_preencode_encode_graph_compute_ms\":{:.6f},"
        "\"tail_encode_graph_compute_ms\":{:.6f},"
        "\"tail_propagate_graph_compute_ms\":{:.6f},"
        "\"tail_propagate_cache_compute_ms\":{:.6f},"
        "\"tail_propagate_input_prompt_upload_ms\":{:.6f},"
        "\"tail_propagate_input_rope_upload_ms\":{:.6f},"
        "\"tail_propagate_input_memory_upload_ms\":{:.6f},"
        "\"tail_propagate_input_feature_upload_ms\":{:.6f},"
        "\"tail_propagate_input_constant_upload_ms\":{:.6f},"
        "\"tail_propagate_input_sparse_upload_ms\":{:.6f},"
        "\"required_core_compute_ms\":{:.6f},"
        "\"required_non_core_compute_ms\":{:.6f},"
        "\"required_core_compute_fraction\":{:.9f},"
        "\"tail_encode_graph_compute_fraction\":{:.9f},"
        "\"track_encode_ms\":{:.6f},"
        "\"track_encode_total_ms\":{:.6f},"
        "\"track_propagate_ms\":{:.6f},"
        "\"track_propagate_total_ms\":{:.6f},"
        "\"track_unattributed_ms\":{:.6f},"
        "\"track_unattributed_total_ms\":{:.6f},"
        "\"track_artifact_ms\":{:.6f},"
        "\"track_artifact_total_ms\":{:.6f},"
        "\"preencode_avg_ms\":{:.6f},"
        "\"preencode_total_ms\":{:.6f},"
        "\"preencode_state_create_avg_ms\":{:.6f},"
        "\"preencode_state_create_total_ms\":{:.6f},"
        "\"track_frames\":{},"
        "\"track_all_frames\":{},"
        "\"track_split_frames\":{},"
        "\"preencoded_frames\":{},"
        "\"frame0_candidates\":{},"
        "\"frame0_added_instances\":{},"
        "\"text_init_selected_only\":{},"
        "\"rss_mib\":{:.6f},"
        "\"detections\":{}}}",
        index,
        model_json,
        format_size(r.file_size),
        backend_json,
        r.t_load_ms,
        r.t_load_ms,
        r.t_frame0_ms,
        r.t_frame0_encode_ms,
        r.t_frame0_prompt_ms,
        r.t_frame0_segment_ms,
        r.t_frame0_tracker_add_ms,
        r.t_frame0_tracker_mask_prepare_ms,
        r.t_frame0_tracker_memory_encode_ms,
        r.t_frame0_tracker_obj_ptr_ms,
        r.t_frame0_tracker_store_ms,
        r.t_frame0_tracker_unattributed_ms,
        r.t_frame0_prompt_unattributed_ms,
        r.t_track_avg_ms,
        r.t_track_p50_ms,
        r.t_track_p95_ms,
        r.t_total_ms,
        r.t_session_ms,
        r.t_input_load_ms,
        r.t_input_directory_scan_ms,
        r.t_input_sort_ms,
        r.t_input_frame_decode_ms,
        r.t_input_frame_store_ms,
        r.t_input_remainder_ms,
        r.n_input_frames,
        input_source,
        r.t_model_e2e_ms,
        r.t_required_e2e_ms,
        r.t_cold_required_e2e_ms,
        r.t_input_plus_session_ms,
        r.t_required_accounted_ms,
        r.t_required_remainder_ms,
        r.t_required_accounting_delta_ms,
        r.t_model_accounting_delta_ms,
        r.t_model_accounted_ms,
        r.t_session_overhead_ms,
        r.t_model_remainder_ms,
        r.t_state_create_ms,
        r.t_tracker_create_ms,
        r.t_session_setup_ms,
        r.t_artifact_ms,
        r.t_frame0_artifact_ms,
        r.t_frame0_model_ms,
        r.t_frame0_prompt_model_ms,
        r.t_track_all_ms,
        r.t_track_all_model_ms,
        r.t_tail_model_ms,
        r.t_track_all_ms,
        r.t_track_all_model_ms,
        r.t_preencode_state_create_total_ms + r.t_preencode_total_ms,
        r.t_tail_model_ms,
        r.t_tail_encode_required_total_ms,
        r.t_required_tail_state_create_ms,
        r.t_required_tail_encode_ms,
        r.t_required_tail_propagate_ms,
        r.t_required_tail_unattributed_ms,
        r.t_required_output_artifact_ms,
        r.t_tail_inline_encode_ms,
        r.t_tail_preencode_encode_ms,
        r.t_tail_inline_propagate_ms,
        r.t_tail_preencoded_propagate_ms,
        r.t_frame0_encode_graph_compute_ms,
        r.t_tail_inline_encode_graph_compute_ms,
        r.t_tail_preencode_encode_graph_compute_ms,
        r.t_tail_encode_graph_compute_ms,
        r.t_tail_propagate_graph_compute_ms,
        r.t_tail_propagate_cache_compute_ms,
        r.t_tail_propagate_input_prompt_upload_ms,
        r.t_tail_propagate_input_rope_upload_ms,
        r.t_tail_propagate_input_memory_upload_ms,
        r.t_tail_propagate_input_feature_upload_ms,
        r.t_tail_propagate_input_constant_upload_ms,
        r.t_tail_propagate_input_sparse_upload_ms,
        r.t_required_core_compute_ms,
        r.t_required_non_core_compute_ms,
        r.t_required_core_compute_fraction,
        r.t_tail_encode_graph_compute_fraction,
        r.t_track_encode_avg_ms,
        r.t_track_encode_total_ms,
        r.t_track_propagate_avg_ms,
        r.t_track_propagate_total_ms,
        r.t_track_unattributed_avg_ms,
        r.t_track_unattributed_total_ms,
        r.t_track_artifact_avg_ms,
        r.t_track_artifact_total_ms,
        r.t_preencode_avg_ms,
        r.t_preencode_total_ms,
        r.t_preencode_state_create_avg_ms,
        r.t_preencode_state_create_total_ms,
        r.n_track_frames,
        r.n_track_all_frames,
        r.n_track_split_frames,
        r.n_preencoded_frames,
        r.n_frame0_candidates,
        r.n_frame0_added_instances,
        r.text_init_selected_only != 0,
        r.max_rss_kib / 1024.0,
        r.n_detections);
}

static double percentile(std::vector<double> values, double p) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = (values.size() - 1) * p;
    const size_t lo = (size_t) pos;
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = pos - (double) lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static const sam3_detection* select_detection(
    const sam3_result& result, std::optional<size_t> candidate_index = std::nullopt) {
    if (result.detections.empty())
        return nullptr;
    if (candidate_index && *candidate_index < result.detections.size()) {
        return &result.detections[*candidate_index];
    }
    return &*std::ranges::max_element(
        result.detections, {}, [](const sam3_detection& det) { return det.iou_score; });
}

static const sam3_detection* select_detection_by_instance(const sam3_result& result,
                                                          std::optional<int> instance_id) {
    if (!instance_id) {
        return nullptr;
    }
    const auto it = std::ranges::find_if(result.detections, [&](const sam3_detection& det) {
        return det.instance_id == *instance_id;
    });
    return it == result.detections.end() ? nullptr : &*it;
}

static const sam3_detection* select_detection_for_output(
    const sam3_result& result,
    std::optional<size_t> candidate_index = std::nullopt,
    std::optional<int> instance_id = std::nullopt) {
    if (const auto* det = select_detection_by_instance(result, instance_id)) {
        return det;
    }
    return select_detection(result, candidate_index);
}

static std::optional<size_t> selected_detection_index(
    const sam3_result& result, std::optional<size_t> candidate_index = std::nullopt) {
    if (result.detections.empty())
        return std::nullopt;
    if (candidate_index && *candidate_index < result.detections.size()) {
        return candidate_index;
    }
    const auto best = std::ranges::max_element(
        result.detections, {}, [](const sam3_detection& det) { return det.iou_score; });
    return static_cast<size_t>(std::ranges::distance(result.detections.begin(), best));
}

static std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
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

static double encode_timing_accounted_ms(const sam3_encode_timing& timing) {
    return timing.preprocess_ms + timing.graph_build_ms + timing.graph_alloc_ms +
           timing.input_upload_ms + timing.graph_compute_ms + timing.state_update_ms +
           timing.pe_build_ms;
}

static double encode_timing_remainder_ms(const sam3_encode_timing& timing, double measured_ms) {
    return std::max(0.0, measured_ms - encode_timing_accounted_ms(timing));
}

static void write_encode_timing_fields(std::ostream& out,
                                       const sam3_encode_timing& timing,
                                       double measured_ms) {
    out << std::format(
        ",\"encode_timing_total_ms\":{:.6f},"
        "\"encode_preprocess_ms\":{:.6f},"
        "\"encode_graph_build_ms\":{:.6f},"
        "\"encode_graph_alloc_ms\":{:.6f},"
        "\"encode_input_upload_ms\":{:.6f},"
        "\"encode_graph_compute_ms\":{:.6f},"
        "\"encode_state_update_ms\":{:.6f},"
        "\"encode_pe_build_ms\":{:.6f},"
        "\"encode_accounted_ms\":{:.6f},"
        "\"encode_remainder_ms\":{:.6f},"
        "\"encode_used_cuda_preprocess\":{},"
        "\"encode_include_detector_neck\":{}",
        timing.total_ms,
        timing.preprocess_ms,
        timing.graph_build_ms,
        timing.graph_alloc_ms,
        timing.input_upload_ms,
        timing.graph_compute_ms,
        timing.state_update_ms,
        timing.pe_build_ms,
        encode_timing_accounted_ms(timing),
        encode_timing_remainder_ms(timing, measured_ms),
        timing.used_cuda_preprocess,
        timing.include_detector_neck);
}

static double propagate_timing_accounted_ms(const sam3_propagate_timing& timing) {
    return timing.prepare_caches_ms + timing.memory_slot_read_ms + timing.obj_ptr_read_ms +
           timing.prompt_build_ms + timing.memory_prepare_ms + timing.ptr_prepare_ms +
           timing.rope_cache_ms + timing.rope_k_build_ms + timing.graph_build_ms +
           timing.graph_alloc_ms + timing.input_upload_ms + timing.graph_compute_ms +
           timing.output_read_ms + timing.graph_cache_build_ms +
           timing.graph_cache_input_upload_ms + timing.graph_cache_compute_ms +
           timing.graph_cache_output_read_ms + timing.bbox_from_logits_ms +
           timing.fullmask_active_resize_bbox_ms + timing.fullmask_pending_resize_bbox_ms +
           timing.memory_update_ms + timing.tracker_update_ms + timing.result_build_ms;
}

static double propagate_timing_remainder_ms(const sam3_propagate_timing& timing,
                                            double measured_ms) {
    return std::max(0.0, measured_ms - propagate_timing_accounted_ms(timing));
}

static double input_timing_accounted_ms(const BenchWire& session) {
    return session.t_input_directory_scan_ms + session.t_input_sort_ms +
           session.t_input_frame_decode_ms + session.t_input_frame_store_ms;
}

static void write_propagate_timing_fields(std::ostream& out,
                                          const sam3_propagate_timing& timing,
                                          double measured_ms) {
    out << std::format(
        ",\"propagate_timing_total_ms\":{:.6f},"
        "\"propagate_prepare_caches_ms\":{:.6f},"
        "\"propagate_memory_slot_read_ms\":{:.6f},"
        "\"propagate_obj_ptr_read_ms\":{:.6f},"
        "\"propagate_prompt_build_ms\":{:.6f},"
        "\"propagate_memory_prepare_ms\":{:.6f},"
        "\"propagate_ptr_prepare_ms\":{:.6f},"
        "\"propagate_rope_cache_ms\":{:.6f},"
        "\"propagate_rope_k_build_ms\":{:.6f},"
        "\"propagate_graph_build_ms\":{:.6f},"
        "\"propagate_graph_alloc_ms\":{:.6f},"
        "\"propagate_input_upload_ms\":{:.6f},"
        "\"propagate_input_prompt_upload_ms\":{:.6f},"
        "\"propagate_input_rope_upload_ms\":{:.6f},"
        "\"propagate_input_memory_upload_ms\":{:.6f},"
        "\"propagate_input_feature_upload_ms\":{:.6f},"
        "\"propagate_input_constant_upload_ms\":{:.6f},"
        "\"propagate_input_sparse_upload_ms\":{:.6f},"
        "\"propagate_graph_compute_ms\":{:.6f},"
        "\"propagate_output_read_ms\":{:.6f},"
        "\"propagate_graph_cache_build_ms\":{:.6f},"
        "\"propagate_graph_cache_input_upload_ms\":{:.6f},"
        "\"propagate_graph_cache_compute_ms\":{:.6f},"
        "\"propagate_graph_cache_output_read_ms\":{:.6f},"
        "\"propagate_bbox_from_logits_ms\":{:.6f},"
        "\"propagate_fullmask_active_resize_bbox_ms\":{:.6f},"
        "\"propagate_fullmask_pending_resize_bbox_ms\":{:.6f},"
        "\"propagate_memory_update_ms\":{:.6f},"
        "\"propagate_tracker_update_ms\":{:.6f},"
        "\"propagate_result_build_ms\":{:.6f},"
        "\"propagate_accounted_ms\":{:.6f},"
        "\"propagate_remainder_ms\":{:.6f},"
        "\"propagate_single_calls\":{},"
        "\"propagate_active_masklets\":{},"
        "\"propagate_pending_masklets\":{},"
        "\"propagate_used_sam31_graph_cache\":{},"
        "\"propagate_aliased_feature_inputs\":{},"
        "\"propagate_aliased_constant_inputs\":{},"
        "\"propagate_aliased_feature_input_count\":{},"
        "\"propagate_aliased_constant_input_count\":{},"
        "\"propagate_uploaded_feature_input_count\":{},"
        "\"propagate_uploaded_constant_input_count\":{}",
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
        propagate_timing_accounted_ms(timing),
        propagate_timing_remainder_ms(timing, measured_ms),
        timing.single_calls,
        timing.active_masklets,
        timing.pending_masklets,
        timing.used_sam31_graph_cache,
        timing.aliased_feature_inputs,
        timing.aliased_constant_inputs,
        timing.aliased_feature_input_count,
        timing.aliased_constant_input_count,
        timing.uploaded_feature_input_count,
        timing.uploaded_constant_input_count);
}

static void write_metadata_row(std::ostream* out,
                               const std::string& model_path,
                               std::string_view backend,
                               const std::string& video_path,
                               const std::string& frame_dir,
                               const sam3_image& first_frame,
                               int n_frames,
                               float px,
                               float py,
                               int requested_encode_img_size,
                               int effective_encode_img_size,
                               bool bbox_only,
                               bool multimask,
                               std::string_view text_prompt,
                               bool preencode_track_frames,
                               bool preencode_cached_tail_frames,
                               int timed_start_frame) {
    if (!out)
        return;
    const std::string model_path_json = json_escape(model_path);
    const std::string backend_json = json_escape(backend);
    const std::string video_path_json = json_escape(video_path);
    const std::string frame_dir_json = json_escape(frame_dir);
    const std::string text_prompt_json = json_escape(std::string{text_prompt});
    const std::string input_source = frame_dir.empty() ? "video" : "frame_dir";
    *out << std::format(
        "{{\"source\":\"sam3cpp-meta\","
        "\"model_path\":\"{}\","
        "\"backend\":\"{}\","
        "\"input_source\":\"{}\","
        "\"video_path\":\"{}\","
        "\"frame_dir\":\"{}\","
        "\"decoded_width\":{},\"decoded_height\":{},"
        "\"frames\":{},"
        "\"point_x\":{:.6f},\"point_y\":{:.6f},"
        "\"text_prompt\":\"{}\","
        "\"encode_img_size_requested\":{},"
        "\"encode_img_size_effective\":{},"
        "\"bbox_only\":{},\"multimask\":{},"
        "\"preencoded_track_frames\":{},"
        "\"preencoded_cached_tail_frames\":{},"
        "\"preencoded_track_mode\":\"{}\","
        "\"timed_start_frame\":{}}}\n",
        model_path_json,
        backend_json,
        input_source,
        video_path_json,
        frame_dir_json,
        first_frame.width,
        first_frame.height,
        n_frames,
        px,
        py,
        text_prompt_json,
        requested_encode_img_size,
        effective_encode_img_size,
        bbox_only,
        multimask,
        preencode_track_frames,
        preencode_cached_tail_frames,
        preencode_track_frames ? "all" : (preencode_cached_tail_frames ? "cached_tail" : "none"),
        timed_start_frame);
}

static bool write_frame_timing_rows(const std::string& output_frame_timing_jsonl,
                                    const std::string& model_path,
                                    std::string_view backend,
                                    const BenchWire& session,
                                    const std::vector<FrameTimingRow>& rows) {
    if (output_frame_timing_jsonl.empty()) {
        return true;
    }

    std::ofstream out(output_frame_timing_jsonl);
    if (!out) {
        return false;
    }

    const std::string model_json =
        json_escape(std::filesystem::path{model_path}.filename().string());
    const std::string backend_json = json_escape(backend);
    const std::string_view input_source = session.input_from_frame_dir != 0 ? "frame_dir" : "video";
    out << std::format(
        "{{\"source\":\"sam3cpp-frame-timing\","
        "\"stage\":\"input\","
        "\"model\":\"{}\","
        "\"backend\":\"{}\","
        "\"frame_index\":null,"
        "\"timed_tail\":false,"
        "\"included_in_required_e2e\":true,"
        "\"input_source\":\"{}\","
        "\"input_frames\":{},"
        "\"total_ms\":{:.6f},"
        "\"directory_scan_ms\":{:.6f},"
        "\"sort_ms\":{:.6f},"
        "\"frame_decode_ms\":{:.6f},"
        "\"frame_store_ms\":{:.6f},"
        "\"accounted_ms\":{:.6f},"
        "\"remainder_ms\":{:.6f}}}\n",
        model_json,
        backend_json,
        input_source,
        session.n_input_frames,
        session.t_input_load_ms,
        session.t_input_directory_scan_ms,
        session.t_input_sort_ms,
        session.t_input_frame_decode_ms,
        session.t_input_frame_store_ms,
        input_timing_accounted_ms(session),
        session.t_input_remainder_ms);
    out << std::format(
        "{{\"source\":\"sam3cpp-frame-timing\","
        "\"stage\":\"frame0\","
        "\"model\":\"{}\","
        "\"backend\":\"{}\","
        "\"frame_index\":0,"
        "\"timed_tail\":false,"
        "\"included_in_required_e2e\":true,"
        "\"total_ms\":{:.6f},"
        "\"model_ms\":{:.6f},"
        "\"encode_ms\":{:.6f},"
        "\"prompt_ms\":{:.6f},"
        "\"segment_ms\":{:.6f},"
        "\"tracker_add_ms\":{:.6f},"
        "\"tracker_mask_prepare_ms\":{:.6f},"
        "\"tracker_memory_encode_ms\":{:.6f},"
        "\"tracker_obj_ptr_ms\":{:.6f},"
        "\"tracker_store_ms\":{:.6f},"
        "\"tracker_unattributed_ms\":{:.6f},"
        "\"prompt_unattributed_ms\":{:.6f},"
        "\"artifact_ms\":{:.6f}",
        model_json,
        backend_json,
        session.t_frame0_ms,
        session.t_frame0_model_ms,
        session.t_frame0_encode_ms,
        session.t_frame0_prompt_ms,
        session.t_frame0_segment_ms,
        session.t_frame0_tracker_add_ms,
        session.t_frame0_tracker_mask_prepare_ms,
        session.t_frame0_tracker_memory_encode_ms,
        session.t_frame0_tracker_obj_ptr_ms,
        session.t_frame0_tracker_store_ms,
        session.t_frame0_tracker_unattributed_ms,
        session.t_frame0_prompt_unattributed_ms,
        session.t_frame0_artifact_ms);
    write_encode_timing_fields(out, session.frame0_encode_timing, session.t_frame0_encode_ms);
    out << std::format(
        ",\"candidates\":{},"
        "\"added_instances\":{}}}\n",
        session.n_frame0_candidates,
        session.n_frame0_added_instances);

    for (const auto& row : rows) {
        out << std::format(
            "{{\"source\":\"sam3cpp-frame-timing\","
            "\"stage\":\"tail\","
            "\"model\":\"{}\","
            "\"backend\":\"{}\","
            "\"frame_index\":{},"
            "\"timed_tail\":{},"
            "\"included_in_required_e2e\":true,"
            "\"split_timing\":{},"
            "\"preencoded\":{},"
            "\"detections\":{},"
            "\"total_ms\":{:.6f},"
            "\"model_ms\":{:.6f},"
            "\"runtime_wall_ms\":{:.6f},"
            "\"visible_model_ms\":{:.6f},"
            "\"preencode_required_model_ms\":{:.6f},"
            "\"required_model_ms\":{:.6f},"
            "\"required_wall_ms\":{:.6f},"
            "\"inline_encode_ms\":{:.6f},"
            "\"preencode_state_create_ms\":{:.6f},"
            "\"preencode_ms\":{:.6f},"
            "\"required_encode_ms\":{:.6f},"
            "\"propagate_ms\":{:.6f},"
            "\"artifact_ms\":{:.6f},"
            "\"unattributed_ms\":{:.6f}",
            model_json,
            backend_json,
            row.frame_index,
            row.timed,
            row.split_timing,
            row.preencoded,
            row.detections,
            row.total_ms,
            row.model_ms,
            row.runtime_wall_ms,
            row.visible_model_ms,
            row.preencode_required_model_ms,
            row.required_model_ms,
            row.required_wall_ms,
            row.inline_encode_ms,
            row.preencode_state_create_ms,
            row.preencode_ms,
            row.required_encode_ms,
            row.propagate_ms,
            row.artifact_ms,
            row.unattributed_ms);
        write_encode_timing_fields(out, row.encode_timing, row.required_encode_ms);
        write_propagate_timing_fields(out, row.propagate_timing, row.propagate_ms);
        out << "}\n";
    }
    return static_cast<bool>(out);
}

static std::string json_float_array(const std::vector<float>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::format("{:.6f}", values[i]);
    }
    out += "]";
    return out;
}

static std::string json_int_array(const std::vector<int>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::format("{}", values[i]);
    }
    out += "]";
    return out;
}

static sam3_detection make_visual_point_detection(const sam3_image& image, float px, float py) {
    sam3_detection detection;
    detection.score = 1.0f;
    detection.iou_score = 1.0f;
    detection.obj_score_logit = 10.0f;
    detection.mask.width = image.width;
    detection.mask.height = image.height;
    detection.mask.iou_score = 1.0f;
    detection.mask.obj_score = 1.0f;
    detection.mask.data.assign(static_cast<size_t>(image.width) * static_cast<size_t>(image.height),
                               0);

    const int side = std::max(16, std::min(image.width, image.height) / 5);
    const int cx = std::clamp(static_cast<int>(std::lround(px)), 0, image.width - 1);
    const int cy = std::clamp(static_cast<int>(std::lround(py)), 0, image.height - 1);
    const int x0 = std::clamp(cx - side / 2, 0, image.width - 1);
    const int y0 = std::clamp(cy - side / 2, 0, image.height - 1);
    const int x1 = std::clamp(x0 + side, x0 + 1, image.width);
    const int y1 = std::clamp(y0 + side, y0 + 1, image.height);
    detection.box = {
        .x0 = static_cast<float>(x0),
        .y0 = static_cast<float>(y0),
        .x1 = static_cast<float>(x1),
        .y1 = static_cast<float>(y1),
    };
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            detection.mask.data[static_cast<size_t>(y) * static_cast<size_t>(image.width) +
                                static_cast<size_t>(x)] = 255;
        }
    }
    return detection;
}

static void write_detection_row(std::ostream* out,
                                int offset,
                                int expected_frame_index,
                                const sam3_result& result,
                                std::string_view mask_path = {},
                                std::optional<size_t> candidate_index = std::nullopt,
                                std::optional<int> instance_id = std::nullopt) {
    if (!out)
        return;
    if (result.detections.empty()) {
        *out << std::format(
            "{{\"offset\":{},\"expected_frame_index\":{},\"bbox_xyxy\":null,"
            "\"score\":0,\"mask_area\":0,\"mask_fnv1a64\":\"0000000000000000\","
            "\"mask_path\":null,"
            "\"source\":\"sam3cpp-missing\"}}\n",
            offset,
            expected_frame_index);
        return;
    }
    const auto* best = select_detection_for_output(result, candidate_index, instance_id);
    if (!best)
        return;
    const auto& det = *best;
    uint64_t mask_hash = 1469598103934665603ULL;
    int mask_area = 0;
    for (uint8_t v : det.mask.data) {
        if (v > 127)
            mask_area++;
        mask_hash ^= (uint64_t) v;
        mask_hash *= 1099511628211ULL;
    }
    const std::string mask_path_json =
        mask_path.empty() ? "null" : std::format("\"{}\"", json_escape(mask_path));
    const std::string decoder_iou_scores_json = json_float_array(det.decoder_iou_scores);
    const std::string decoder_lowres_mask_areas_json =
        json_int_array(det.decoder_lowres_mask_areas);
    *out << std::format(
        "{{\"offset\":{},\"expected_frame_index\":{},"
        "\"bbox_xyxy\":[{:.3f},{:.3f},{:.3f},{:.3f}],\"score\":{:.6f},"
        "\"selected_mask_index\":{},\"decoder_iou_scores\":{},"
        "\"decoder_lowres_mask_areas\":{},"
        "\"mask_area\":{},\"mask_fnv1a64\":\"{:016x}\","
        "\"mask_path\":{},"
        "\"source\":\"sam3cpp-edgetam\"}}\n",
        offset,
        expected_frame_index,
        det.box.x0,
        det.box.y0,
        det.box.x1,
        det.box.y1,
        det.score,
        det.selected_mask_index,
        decoder_iou_scores_json,
        decoder_lowres_mask_areas_json,
        mask_area,
        mask_hash,
        mask_path_json);
}

static void write_initial_candidate_rows(std::ostream* out,
                                         const sam3_result& result,
                                         std::optional<size_t> selected_index = std::nullopt) {
    if (!out)
        return;
    for (size_t candidate_index = 0; candidate_index < result.detections.size();
         ++candidate_index) {
        const auto& det = result.detections[candidate_index];
        uint64_t mask_hash = 1469598103934665603ULL;
        int mask_area = 0;
        for (uint8_t v : det.mask.data) {
            if (v > 127)
                mask_area++;
            mask_hash ^= (uint64_t) v;
            mask_hash *= 1099511628211ULL;
        }
        *out << std::format(
            "{{\"source\":\"sam3cpp-initial-candidate\","
            "\"frame_index\":0,"
            "\"candidate_index\":{},"
            "\"selected\":{},"
            "\"instance_id\":{},"
            "\"bbox_xyxy\":[{:.3f},{:.3f},{:.3f},{:.3f}],"
            "\"score\":{:.6f},"
            "\"iou_score\":{:.6f},"
            "\"obj_score\":{:.6f},"
            "\"mask_area\":{},"
            "\"mask_fnv1a64\":\"{:016x}\"}}\n",
            candidate_index,
            selected_index && *selected_index == candidate_index,
            det.instance_id,
            det.box.x0,
            det.box.y0,
            det.box.x1,
            det.box.y1,
            det.score,
            det.iou_score,
            det.mask.obj_score,
            mask_area,
            mask_hash);
    }
}

static std::string save_detection_mask(std::string_view output_mask_dir,
                                       int offset,
                                       const sam3_result& result,
                                       std::optional<size_t> candidate_index = std::nullopt,
                                       std::optional<int> instance_id = std::nullopt) {
    const auto* det = select_detection_for_output(result, candidate_index, instance_id);
    if (output_mask_dir.empty() || !det || det->mask.data.empty()) {
        return {};
    }

    std::filesystem::create_directories(std::filesystem::path{output_mask_dir});
    const auto path =
        std::filesystem::path{output_mask_dir} / std::format("frame_{:05d}.png", offset);
    if (!sam3_save_mask(det->mask, path.string())) {
        return {};
    }
    return path.string();
}

static std::string save_detection_logits(std::string_view output_logits_dir,
                                         int offset,
                                         const sam3_result& result,
                                         std::optional<size_t> candidate_index = std::nullopt,
                                         std::optional<int> instance_id = std::nullopt) {
    const auto* det = select_detection_for_output(result, candidate_index, instance_id);
    if (output_logits_dir.empty() || !det || det->raw_mask_logits.empty() ||
        det->raw_mask_width <= 0 || det->raw_mask_height <= 0) {
        return {};
    }

    std::filesystem::create_directories(std::filesystem::path{output_logits_dir});
    const auto base =
        std::filesystem::path{output_logits_dir} / std::format("frame_{:05d}", offset);

    std::ofstream data(base.string() + ".bin", std::ios::binary);
    if (!data) {
        return {};
    }
    data.write(reinterpret_cast<const char*>(det->raw_mask_logits.data()),
               static_cast<std::streamsize>(det->raw_mask_logits.size() * sizeof(float)));
    if (!data) {
        return {};
    }

    std::ofstream shape(base.string() + ".shape");
    if (!shape) {
        return {};
    }
    shape << det->raw_mask_height << " " << det->raw_mask_width << "\n";
    return base.string() + ".bin";
}

// Sort key: family → size → precision
static int model_sort_key(const std::string& name) {
    int family = 0;
    if (name.find("sam3-visual") != std::string::npos)
        family = 1;
    else if (name.find("sam3") != std::string::npos)
        family = 0;
    else if (name.find("sam2.1") != std::string::npos)
        family = 3;
    else if (name.find("sam2") != std::string::npos)
        family = 2;

    int size = 0;
    if (name.find("tiny") != std::string::npos)
        size = 0;
    else if (name.find("small") != std::string::npos)
        size = 1;
    else if (name.find("base_plus") != std::string::npos)
        size = 2;
    else if (name.find("large") != std::string::npos)
        size = 3;

    int prec = 0;
    if (name.find("f32") != std::string::npos)
        prec = 0;
    else if (name.find("f16") != std::string::npos)
        prec = 1;
    else if (name.find("q8_0") != std::string::npos)
        prec = 2;
    else if (name.find("q4_1") != std::string::npos)
        prec = 3;
    else if (name.find("q4_0") != std::string::npos)
        prec = 4;

    return family * 10000 + size * 100 + prec;
}

// ── Model discovery ─────────────────────────────────────────────────────────

struct ModelEntry {
    std::string path;
    std::string name;
    int64_t file_size;
};

static std::vector<ModelEntry> discover_models(const std::string& dir, const std::string& filter) {
    std::vector<ModelEntry> entries;

#ifdef _WIN32
    std::string pattern = dir + "\\*.ggml";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        std::print(stderr, "ERROR: cannot open models directory '{}'\n", dir);
        return entries;
    }
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            continue;
        std::string fname = fd.cFileName;
        if (!ends_with(fname, ".ggml"))
            continue;
        if (!filter.empty() && fname.find(filter) == std::string::npos)
            continue;

        std::string full_path = dir + "\\" + fname;
        struct _stat st;
        if (_stat(full_path.c_str(), &st) != 0)
            continue;

        ModelEntry e;
        e.path = full_path;
        e.name = strip_extension(fname);
        e.file_size = st.st_size;
        entries.push_back(e);
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
#else
    auto opened_dir = DirHandle::open(dir);
    if (!opened_dir) {
        std::print(stderr, "ERROR: cannot open models directory '{}'\n", dir);
        return entries;
    }
    DirHandle d = std::move(*opened_dir);
    struct dirent* ent;
    while ((ent = readdir(d.get())) != nullptr) {
        std::string fname = ent->d_name;
        if (!ends_with(fname, ".ggml"))
            continue;
        if (!filter.empty() && fname.find(filter) == std::string::npos)
            continue;

        std::string full_path = dir + "/" + fname;
        struct stat st;
        if (stat(full_path.c_str(), &st) != 0)
            continue;

        ModelEntry e;
        e.path = full_path;
        e.name = strip_extension(fname);
        e.file_size = st.st_size;
        entries.push_back(e);
    }
#endif

    std::sort(entries.begin(), entries.end(), [](const ModelEntry& a, const ModelEntry& b) {
        return model_sort_key(a.name) < model_sort_key(b.name);
    });
    return entries;
}

// ── Run a single benchmark (shared logic) ──────────────────────────────────

static BenchWire run_single_benchmark(const std::string& model_path,
                                      bool use_gpu,
                                      const std::string& video_path,
                                      const std::string& frame_dir,
                                      int n_frames,
                                      float px,
                                      float py,
                                      const std::string& text_prompt,
                                      int n_threads,
                                      int encode_img_size,
                                      bool bbox_only,
                                      bool multimask,
                                      std::optional<size_t> initial_candidate_index,
                                      int recondition_every,
                                      const std::string& output_jsonl,
                                      const std::string& output_initial_candidates_jsonl,
                                      const std::string& output_frame_timing_jsonl,
                                      const std::string& output_mask_dir,
                                      const std::string& output_logits_dir,
                                      int warmup_runs,
                                      bool preencode_track_frames_requested,
                                      bool preencode_cached_tail_frames_requested,
                                      bool text_init_selected_only,
                                      int timed_start_frame,
                                      bool quiet) {
    BenchWire wire = {};

    auto set_wire_string = [](auto& dst, std::string_view value) {
        const size_t n = std::min(dst.size() - 1, value.size());
        std::ranges::copy(value.substr(0, n), dst.begin());
        dst[n] = '\0';
    };

    auto fail = [&](std::string_view msg) {
        wire.ok = 0;
        set_wire_string(wire.error, msg);
    };

    std::ofstream out;
    if (!output_jsonl.empty()) {
        out.open(output_jsonl);
        if (!out) {
            fail("open output_jsonl failed");
            return wire;
        }
    }

    std::ofstream candidate_out;
    if (!output_initial_candidates_jsonl.empty()) {
        candidate_out.open(output_initial_candidates_jsonl);
        if (!candidate_out) {
            fail("open initial candidates JSONL failed");
            return wire;
        }
    }

    if (!output_logits_dir.empty()) {
#ifdef _WIN32
        _putenv_s("SAM3_CAPTURE_PROP_LOGITS", "1");
#else
        setenv("SAM3_CAPTURE_PROP_LOGITS", "1", 1);
#endif
    }

    const int64_t input_load_t0 = ggml_time_us();
    auto frame_result = load_benchmark_frames(video_path, frame_dir, n_frames);
    wire.t_input_load_ms = (ggml_time_us() - input_load_t0) / 1000.0;
    if (!frame_result) {
        fail(frame_result.error());
        return wire;
    }
    const auto input_timing = frame_result->timing;
    wire.t_input_directory_scan_ms = input_timing.directory_scan_ms;
    wire.t_input_sort_ms = input_timing.sort_ms;
    wire.t_input_frame_decode_ms = input_timing.frame_decode_ms;
    wire.t_input_frame_store_ms = input_timing.frame_store_ms;
    wire.n_input_frames = input_timing.frames;
    wire.input_from_frame_dir = input_timing.from_frame_dir ? 1 : 0;
    wire.t_input_remainder_ms =
        std::max(0.0,
                 wire.t_input_load_ms - wire.t_input_directory_scan_ms - wire.t_input_sort_ms -
                     wire.t_input_frame_decode_ms - wire.t_input_frame_store_ms);
    std::vector<sam3_image> frames = std::move(frame_result->frames);

    // Load model
    int64_t t0 = ggml_time_us();

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = use_gpu;
    params.require_gpu = use_gpu;
    params.n_threads = n_threads;
    params.encode_img_size = encode_img_size;

    auto model = sam3_load_model(params);
    if (!model) {
        fail("load failed");
        return wire;
    }
    set_wire_string(wire.backend, sam3_backend_name(*model));
    const std::string_view actual_backend = sam3_backend_name(*model);
    if (use_gpu && !actual_backend.starts_with("CUDA") && !actual_backend.starts_with("Metal")) {
        fail(std::format("requested GPU backend unavailable; loaded {}", actual_backend));
        return wire;
    }
    const bool model_visual_only = sam3_is_visual_only(*model);
    const bool model_sam3_text_init = !model_visual_only &&
                                      sam3_get_model_type(*model) == SAM3_MODEL_SAM3 &&
                                      !text_prompt.empty();
    const bool effective_preencode_track_frames =
        (preencode_track_frames_requested ||
         (std::getenv("SAM3_BENCH_PREENCODE_TRACK_FRAMES") != nullptr &&
          std::getenv("SAM3_BENCH_PREENCODE_TRACK_FRAMES")[0] != '0')) &&
        (model_visual_only || model_sam3_text_init);
    const bool effective_preencode_cached_tail_frames =
        !effective_preencode_track_frames &&
        (preencode_cached_tail_frames_requested ||
         (std::getenv("SAM3_BENCH_PREENCODE_CACHED_TAIL_FRAMES") != nullptr &&
          std::getenv("SAM3_BENCH_PREENCODE_CACHED_TAIL_FRAMES")[0] != '0')) &&
        (model_visual_only || model_sam3_text_init);
    if (out.is_open()) {
        const int effective_encode_img_size =
            (encode_img_size > 0) ? encode_img_size : sam3_model_image_size(*model);
        write_metadata_row(&out,
                           model_path,
                           sam3_backend_name(*model),
                           video_path,
                           frame_dir,
                           frames[0],
                           n_frames,
                           px,
                           py,
                           encode_img_size,
                           effective_encode_img_size,
                           bbox_only,
                           multimask,
                           text_prompt,
                           effective_preencode_track_frames,
                           effective_preencode_cached_tail_frames,
                           timed_start_frame);
    }

    wire.t_load_ms = (ggml_time_us() - t0) / 1000.0;

    const auto run_session = [&](bool write_outputs,
                                 bool session_quiet,
                                 std::vector<FrameTimingRow>* frame_timings = nullptr) {
        const int64_t session_wall_t0 = ggml_time_us();
        BenchWire session = {};
        session.text_init_selected_only = text_init_selected_only ? 1 : 0;
        if (frame_timings) {
            frame_timings->clear();
            frame_timings->reserve(static_cast<size_t>(std::max(0, n_frames - 1)));
        }
        auto fail_session = [&](std::string_view msg) {
            session.ok = 0;
            set_wire_string(session.error, msg);
        };
        auto record_last_add_timing = [&] {
            const auto timing = sam3_tracker_last_add_timing();
            session.t_frame0_tracker_mask_prepare_ms += timing.mask_prepare_ms;
            session.t_frame0_tracker_memory_encode_ms += timing.memory_encode_ms;
            session.t_frame0_tracker_obj_ptr_ms += timing.obj_ptr_ms;
            session.t_frame0_tracker_store_ms += timing.store_ms;
        };
        auto record_frame0_artifact = [&](int64_t artifact_t0) {
            const double artifact_ms = (ggml_time_us() - artifact_t0) / 1000.0;
            session.t_artifact_ms += artifact_ms;
            session.t_frame0_artifact_ms += artifact_ms;
        };

        const int64_t state_create_t0 = ggml_time_us();
        auto state = sam3_create_state(*model, params);
        session.t_state_create_ms = (ggml_time_us() - state_create_t0) / 1000.0;
        if (!state) {
            fail_session("state failed");
            return session;
        }

        const bool visual_only = model_visual_only;
        const bool sam3_text_init = model_sam3_text_init;
        const bool preencode_track_frames = effective_preencode_track_frames;
        const bool preencode_cached_tail_frames = effective_preencode_cached_tail_frames;
        sam3_tracker_ptr tracker;

        const int64_t tracker_create_t0 = ggml_time_us();
        if (visual_only) {
            sam3_visual_track_params vtp;
            vtp.max_keep_alive = 100;
            vtp.recondition_every = recondition_every;
            vtp.bbox_only = bbox_only;
            tracker = sam3_create_visual_tracker(*model, vtp);
        } else {
            sam3_video_params vp;
            vp.text_prompt = sam3_text_init ? std::string{} : text_prompt;
            vp.hotstart_delay = 0;
            vp.max_keep_alive = 100;
            vp.recondition_every = recondition_every;
            vp.bbox_only = bbox_only;
            tracker = sam3_create_tracker(*model, vp);
        }
        session.t_tracker_create_ms = (ggml_time_us() - tracker_create_t0) / 1000.0;
        session.t_session_setup_ms = session.t_state_create_ms + session.t_tracker_create_ms;
        if (!tracker) {
            fail_session("tracker failed");
            return session;
        }

        if (preencode_track_frames_requested && !(visual_only || sam3_text_init)) {
            fail_session("--preencode-track-frames requires SAM3 text init or visual tracker mode");
            return session;
        }
        if (preencode_cached_tail_frames_requested && !(visual_only || sam3_text_init)) {
            fail_session(
                "--preencode-cached-tail-frames requires SAM3 text init or visual tracker mode");
            return session;
        }

        // Frame 0: encode + add instance
        int64_t session_t0 = ggml_time_us();
        std::optional<int> target_instance_id;

        if (!sam3_encode_image(*state, *model, frames[0])) {
            fail_session("encode f0 failed");
            return session;
        }
        session.frame0_encode_timing = sam3_last_encode_timing();
        session.t_frame0_encode_ms = (ggml_time_us() - session_t0) / 1000.0;
        session.t_frame0_encode_graph_compute_ms = session.frame0_encode_timing.graph_compute_ms;
        session_t0 = ggml_time_us();

        if (visual_only) {
            auto detection = make_visual_point_detection(frames[0], px, py);
            const int64_t add_t0 = ggml_time_us();
            const int inst_id = sam3_tracker_add_detection(*tracker, *state, *model, detection);
            session.t_frame0_tracker_add_ms += (ggml_time_us() - add_t0) / 1000.0;
            record_last_add_timing();
            session.n_frame0_candidates = 1;
            session.n_frame0_added_instances = inst_id >= 0 ? 1 : 0;
            if (inst_id < 0) {
                fail_session("add visual detection failed");
                return session;
            }
            target_instance_id = inst_id;
            detection.instance_id = inst_id;
            detection.mask.instance_id = inst_id;
            sam3_result first;
            first.detections.push_back(std::move(detection));
            const std::optional<size_t> selected_index{0};
            if (write_outputs && out.is_open()) {
                const int64_t artifact_t0 = ggml_time_us();
                const auto mask_path =
                    save_detection_mask(output_mask_dir, 0, first, selected_index);
                save_detection_logits(output_logits_dir, 0, first, selected_index);
                write_detection_row(&out, 0, 0, first, mask_path, selected_index, inst_id);
                record_frame0_artifact(artifact_t0);
            }
            if (write_outputs && candidate_out.is_open()) {
                const int64_t artifact_t0 = ggml_time_us();
                write_initial_candidate_rows(&candidate_out, first, selected_index);
                record_frame0_artifact(artifact_t0);
            }
        } else if (sam3_text_init) {
            sam3_pcs_params pcs;
            pcs.text_prompt = text_prompt;
            int64_t segment_t0 = ggml_time_us();
            sam3_result first = sam3_segment_pcs(*state, *model, pcs);
            session.t_frame0_segment_ms += (ggml_time_us() - segment_t0) / 1000.0;
            if (first.detections.empty()) {
                pcs.score_threshold = -std::numeric_limits<float>::infinity();
                segment_t0 = ggml_time_us();
                first = sam3_segment_pcs(*state, *model, pcs);
                session.t_frame0_segment_ms += (ggml_time_us() - segment_t0) / 1000.0;
                if (first.detections.empty()) {
                    fail_session("text init returned no detections");
                    return session;
                }
                const auto best = std::ranges::max_element(
                    first.detections, {}, [](const sam3_detection& detection) {
                        return detection.score;
                    });
                first.detections = {*best};
            }
            const auto selected_index = selected_detection_index(first);
            if (write_outputs && out.is_open()) {
                const int64_t artifact_t0 = ggml_time_us();
                const auto mask_path =
                    save_detection_mask(output_mask_dir, 0, first, selected_index);
                save_detection_logits(output_logits_dir, 0, first, selected_index);
                write_detection_row(&out, 0, 0, first, mask_path, selected_index);
                record_frame0_artifact(artifact_t0);
            }
            if (write_outputs && candidate_out.is_open()) {
                const int64_t artifact_t0 = ggml_time_us();
                write_initial_candidate_rows(&candidate_out, first, selected_index);
                record_frame0_artifact(artifact_t0);
            }
            session.n_frame0_candidates = static_cast<int>(first.detections.size());
            auto add_text_detection = [&](size_t det_index) -> bool {
                const auto& det = first.detections[det_index];
                const int64_t add_t0 = ggml_time_us();
                int inst_id = sam3_tracker_add_detection(*tracker, *state, *model, det);
                session.t_frame0_tracker_add_ms += (ggml_time_us() - add_t0) / 1000.0;
                record_last_add_timing();
                if (inst_id < 0) {
                    fail_session("add text detection failed");
                    return false;
                }
                ++session.n_frame0_added_instances;
                if (selected_index && *selected_index == det_index) {
                    target_instance_id = inst_id;
                }
                return true;
            };
            if (text_init_selected_only && selected_index) {
                if (!add_text_detection(*selected_index)) {
                    return session;
                }
            } else {
                for (size_t det_index = 0; det_index < first.detections.size(); ++det_index) {
                    if (!add_text_detection(det_index)) {
                        return session;
                    }
                }
            }
        } else {
            sam3_pvs_params pvs;
            pvs.pos_points.push_back({px, py});
            pvs.multimask = multimask;
            pvs.candidate_index = initial_candidate_index;

            if (write_outputs && (out.is_open() || candidate_out.is_open())) {
                const int64_t artifact_t0 = ggml_time_us();
                sam3_result first = sam3_segment_pvs(*state, *model, pvs);
                const auto selected_index =
                    selected_detection_index(first, initial_candidate_index);
                if (out.is_open()) {
                    const auto mask_path =
                        save_detection_mask(output_mask_dir, 0, first, initial_candidate_index);
                    save_detection_logits(output_logits_dir, 0, first, initial_candidate_index);
                    write_detection_row(&out, 0, 0, first, mask_path, initial_candidate_index);
                }
                write_initial_candidate_rows(
                    candidate_out.is_open() ? &candidate_out : nullptr, first, selected_index);
                record_frame0_artifact(artifact_t0);
            }

            const int64_t add_t0 = ggml_time_us();
            int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
            session.t_frame0_tracker_add_ms += (ggml_time_us() - add_t0) / 1000.0;
            session.n_frame0_candidates = 1;
            session.n_frame0_added_instances = inst_id >= 0 ? 1 : 0;
            if (inst_id < 0) {
                fail_session("add_instance failed");
                return session;
            }
            target_instance_id = inst_id;
        }

        session.t_frame0_prompt_ms = (ggml_time_us() - session_t0) / 1000.0;
        session.t_frame0_ms = session.t_frame0_encode_ms + session.t_frame0_prompt_ms;

        std::vector<sam3_state_ptr> preencoded_states;
        std::vector<double> preencode_state_create_ms_by_frame(static_cast<size_t>(n_frames), 0.0);
        std::vector<double> preencode_ms_by_frame(static_cast<size_t>(n_frames), 0.0);
        std::vector<sam3_encode_timing> preencode_timing_by_frame(static_cast<size_t>(n_frames));
        if (preencode_track_frames || preencode_cached_tail_frames) {
            preencoded_states.resize(static_cast<size_t>(n_frames));
            const int first_preencoded_frame = preencode_track_frames ? 1 : 2;
            for (int f = first_preencoded_frame; f < n_frames; ++f) {
                const int64_t preencode_state_create_t0 = ggml_time_us();
                auto encoded = sam3_create_state(*model, params);
                const double preencode_state_create_ms =
                    (ggml_time_us() - preencode_state_create_t0) / 1000.0;
                session.t_preencode_state_create_total_ms += preencode_state_create_ms;
                preencode_state_create_ms_by_frame[static_cast<size_t>(f)] =
                    preencode_state_create_ms;
                if (!encoded) {
                    fail_session("preencode state failed");
                    return session;
                }
                const int64_t preencode_t0 = ggml_time_us();
                if (!sam3_encode_image_for_tracking(*encoded, *model, frames[f])) {
                    fail_session("preencode frame failed");
                    return session;
                }
                preencode_timing_by_frame[static_cast<size_t>(f)] = sam3_last_encode_timing();
                session.t_tail_preencode_encode_graph_compute_ms +=
                    preencode_timing_by_frame[static_cast<size_t>(f)].graph_compute_ms;
                const double preencode_ms = (ggml_time_us() - preencode_t0) / 1000.0;
                session.t_preencode_total_ms += preencode_ms;
                preencode_ms_by_frame[static_cast<size_t>(f)] = preencode_ms;
                session.n_preencoded_frames++;
                preencoded_states[static_cast<size_t>(f)] = std::move(encoded);
            }
            if (session.n_preencoded_frames > 0) {
                session.t_preencode_avg_ms =
                    session.t_preencode_total_ms / session.n_preencoded_frames;
                session.t_preencode_state_create_avg_ms =
                    session.t_preencode_state_create_total_ms / session.n_preencoded_frames;
            }
        }

        // Frames 1..N-1: track / propagate
        double t_track_sum = 0.0;
        double t_track_all_sum = 0.0;
        double t_track_encode_sum = 0.0;
        double t_track_propagate_sum = 0.0;
        double t_track_unattributed_sum = 0.0;
        double t_track_artifact_sum = 0.0;
        double t_track_all_artifact_sum = 0.0;
        std::vector<double> track_times;
        sam3_result last_result;

        for (int f = 1; f < n_frames; f++) {
            session_t0 = ggml_time_us();
            double encode_ms = 0.0;
            double propagate_ms = 0.0;
            double artifact_ms = 0.0;
            bool split_timing = false;
            bool used_preencoded_state = false;
            sam3_encode_timing encode_timing;
            sam3_propagate_timing propagate_timing;

            if (visual_only || sam3_text_init) {
                if (preencoded_states.size() > static_cast<size_t>(f) &&
                    preencoded_states[static_cast<size_t>(f)]) {
                    used_preencoded_state = true;
                    encode_timing = preencode_timing_by_frame[static_cast<size_t>(f)];
                    const int64_t propagate_t0 = ggml_time_us();
                    last_result = sam3_propagate_encoded_frame(
                        *tracker, *preencoded_states[static_cast<size_t>(f)], *model);
                    propagate_timing = sam3_last_propagate_timing();
                    propagate_ms = (ggml_time_us() - propagate_t0) / 1000.0;
                    split_timing = true;
                } else {
                    const int64_t encode_t0 = ggml_time_us();
                    if (!sam3_encode_image_for_tracking(*state, *model, frames[f])) {
                        fail_session("encode tracking frame failed");
                        return session;
                    }
                    encode_timing = sam3_last_encode_timing();
                    encode_ms = (ggml_time_us() - encode_t0) / 1000.0;
                    const int64_t propagate_t0 = ggml_time_us();
                    last_result = sam3_propagate_encoded_frame(*tracker, *state, *model);
                    propagate_timing = sam3_last_propagate_timing();
                    propagate_ms = (ggml_time_us() - propagate_t0) / 1000.0;
                    split_timing = true;
                }
            } else {
                last_result = sam3_track_frame(*tracker, *state, *model, frames[f]);
            }
            if (write_outputs) {
                const int64_t artifact_t0 = ggml_time_us();
                const auto mask_path = save_detection_mask(
                    output_mask_dir, f, last_result, std::nullopt, target_instance_id);
                save_detection_logits(
                    output_logits_dir, f, last_result, std::nullopt, target_instance_id);
                write_detection_row(out.is_open() ? &out : nullptr,
                                    f,
                                    f,
                                    last_result,
                                    mask_path,
                                    std::nullopt,
                                    target_instance_id);
                artifact_ms = (ggml_time_us() - artifact_t0) / 1000.0;
                session.t_artifact_ms += artifact_ms;
            }

            double dt = (ggml_time_us() - session_t0) / 1000.0;
            const double unattributed_ms =
                split_timing ? std::max(0.0, dt - encode_ms - propagate_ms - artifact_ms) : 0.0;
            const double preencode_state_create_ms =
                preencode_state_create_ms_by_frame[static_cast<size_t>(f)];
            const double preencode_ms = preencode_ms_by_frame[static_cast<size_t>(f)];
            const double visible_model_ms = std::max(0.0, dt - artifact_ms);
            const double preencode_required_model_ms = preencode_state_create_ms + preencode_ms;
            const double required_model_ms = visible_model_ms + preencode_required_model_ms;
            const double required_wall_ms = dt + preencode_required_model_ms;
            if (split_timing) {
                if (used_preencoded_state) {
                    session.t_tail_preencoded_propagate_ms += propagate_ms;
                } else {
                    session.t_tail_inline_encode_ms += encode_ms;
                    session.t_tail_inline_encode_graph_compute_ms += encode_timing.graph_compute_ms;
                    session.t_tail_inline_propagate_ms += propagate_ms;
                }
                session.t_tail_propagate_graph_compute_ms += propagate_timing.graph_compute_ms;
                session.t_tail_propagate_cache_compute_ms +=
                    propagate_timing.graph_cache_compute_ms;
                session.t_tail_propagate_input_prompt_upload_ms +=
                    propagate_timing.input_prompt_upload_ms;
                session.t_tail_propagate_input_rope_upload_ms +=
                    propagate_timing.input_rope_upload_ms;
                session.t_tail_propagate_input_memory_upload_ms +=
                    propagate_timing.input_memory_upload_ms;
                session.t_tail_propagate_input_feature_upload_ms +=
                    propagate_timing.input_feature_upload_ms;
                session.t_tail_propagate_input_constant_upload_ms +=
                    propagate_timing.input_constant_upload_ms;
                session.t_tail_propagate_input_sparse_upload_ms +=
                    propagate_timing.input_sparse_upload_ms;
                session.t_required_tail_unattributed_ms += unattributed_ms;
            }
            if (frame_timings) {
                frame_timings->push_back(FrameTimingRow{
                    .frame_index = f,
                    .timed = f >= timed_start_frame,
                    .split_timing = split_timing,
                    .preencoded = used_preencoded_state,
                    .detections = static_cast<int>(last_result.detections.size()),
                    .total_ms = dt,
                    .model_ms = required_model_ms,
                    .runtime_wall_ms = dt,
                    .visible_model_ms = visible_model_ms,
                    .preencode_required_model_ms = preencode_required_model_ms,
                    .required_model_ms = required_model_ms,
                    .required_wall_ms = required_wall_ms,
                    .inline_encode_ms = encode_ms,
                    .preencode_state_create_ms = preencode_state_create_ms,
                    .preencode_ms = preencode_ms,
                    .required_encode_ms = encode_ms + preencode_ms,
                    .propagate_ms = propagate_ms,
                    .artifact_ms = artifact_ms,
                    .unattributed_ms = unattributed_ms,
                    .encode_timing = encode_timing,
                    .propagate_timing = propagate_timing,
                });
            }
            t_track_all_sum += dt;
            t_track_all_artifact_sum += artifact_ms;
            session.n_track_all_frames++;
            if (f >= timed_start_frame) {
                t_track_sum += dt;
                track_times.push_back(dt);
                session.n_track_frames++;
                if (split_timing) {
                    t_track_encode_sum += encode_ms;
                    t_track_propagate_sum += propagate_ms;
                    t_track_unattributed_sum += unattributed_ms;
                    session.n_track_split_frames++;
                }
                t_track_artifact_sum += artifact_ms;
            }

            if (!session_quiet) {
                std::print(
                    stderr,
                    "    frame {}/{}  {:.0f} ms  encode={:.0f} ms  prop={:.0f} ms  ({} det)\n",
                    f,
                    n_frames - 1,
                    dt,
                    encode_ms,
                    propagate_ms,
                    last_result.detections.size());
            }
        }

        if (session.n_track_frames > 0) {
            session.t_track_avg_ms = t_track_sum / session.n_track_frames;
        }
        if (session.n_track_split_frames > 0) {
            session.t_track_encode_avg_ms = t_track_encode_sum / session.n_track_split_frames;
            session.t_track_encode_total_ms = t_track_encode_sum;
            session.t_track_propagate_avg_ms = t_track_propagate_sum / session.n_track_split_frames;
            session.t_track_propagate_total_ms = t_track_propagate_sum;
            session.t_track_unattributed_avg_ms =
                t_track_unattributed_sum / session.n_track_split_frames;
            session.t_track_unattributed_total_ms = t_track_unattributed_sum;
        }
        if (session.n_track_frames > 0) {
            session.t_track_artifact_avg_ms = t_track_artifact_sum / session.n_track_frames;
            session.t_track_artifact_total_ms = t_track_artifact_sum;
        }
        session.t_track_p50_ms = percentile(track_times, 0.50);
        session.t_track_p95_ms = percentile(track_times, 0.95);
        session.t_total_ms = session.t_frame0_ms + t_track_sum;
        session.t_track_all_ms = t_track_all_sum;
        session.t_session_ms = (ggml_time_us() - session_wall_t0) / 1000.0;
        session.t_model_e2e_ms = std::max(0.0, session.t_session_ms - session.t_artifact_ms);
        const double frame0_prompt_model_ms =
            std::max(0.0, session.t_frame0_prompt_ms - session.t_frame0_artifact_ms);
        const double track_all_model_ms =
            std::max(0.0, session.t_track_all_ms - t_track_all_artifact_sum);
        const double frame0_model_ms = session.t_frame0_encode_ms + frame0_prompt_model_ms;
        session.t_tail_preencode_encode_ms = session.t_preencode_total_ms;
        const double tail_encode_required_total_ms =
            session.t_tail_inline_encode_ms + session.t_tail_preencode_encode_ms;
        const double tail_model_ms = track_all_model_ms +
                                     session.t_preencode_state_create_total_ms +
                                     session.t_preencode_total_ms;
        session.t_tail_encode_graph_compute_ms = session.t_tail_inline_encode_graph_compute_ms +
                                                 session.t_tail_preencode_encode_graph_compute_ms;
        session.t_required_core_compute_ms =
            session.t_frame0_encode_graph_compute_ms + session.t_tail_encode_graph_compute_ms +
            session.t_tail_propagate_graph_compute_ms + session.t_tail_propagate_cache_compute_ms;
        session.t_frame0_model_ms = frame0_model_ms;
        session.t_frame0_prompt_model_ms = frame0_prompt_model_ms;
        session.t_track_all_model_ms = track_all_model_ms;
        session.t_tail_model_ms = tail_model_ms;
        session.t_tail_encode_required_total_ms = tail_encode_required_total_ms;
        session.t_required_tail_state_create_ms = session.t_preencode_state_create_total_ms;
        session.t_required_tail_encode_ms = tail_encode_required_total_ms;
        session.t_required_tail_propagate_ms =
            session.t_tail_inline_propagate_ms + session.t_tail_preencoded_propagate_ms;
        session.t_required_output_artifact_ms = session.t_artifact_ms;
        session.t_frame0_tracker_unattributed_ms =
            std::max(0.0,
                     session.t_frame0_tracker_add_ms - session.t_frame0_tracker_mask_prepare_ms -
                         session.t_frame0_tracker_memory_encode_ms -
                         session.t_frame0_tracker_obj_ptr_ms - session.t_frame0_tracker_store_ms);
        session.t_frame0_prompt_unattributed_ms = std::max(
            0.0,
            frame0_prompt_model_ms - session.t_frame0_segment_ms - session.t_frame0_tracker_add_ms);
        session.t_model_accounted_ms = session.t_session_setup_ms + frame0_model_ms + tail_model_ms;
        session.t_session_overhead_ms =
            std::max(0.0,
                     session.t_session_ms - session.t_session_setup_ms - session.t_frame0_ms -
                         session.t_track_all_ms - session.t_preencode_state_create_total_ms -
                         session.t_preencode_total_ms);
        session.t_model_remainder_ms =
            std::max(0.0, session.t_model_e2e_ms - session.t_model_accounted_ms);
        session.t_model_accounting_delta_ms = session.t_model_e2e_ms - session.t_model_accounted_ms;
        session.t_required_non_core_compute_ms =
            std::max(0.0, session.t_model_e2e_ms - session.t_required_core_compute_ms);
        session.t_required_core_compute_fraction =
            session.t_model_e2e_ms > 0.0
                ? session.t_required_core_compute_ms / session.t_model_e2e_ms
                : 0.0;
        session.t_tail_encode_graph_compute_fraction =
            tail_encode_required_total_ms > 0.0
                ? session.t_tail_encode_graph_compute_ms / tail_encode_required_total_ms
                : 0.0;
        session.n_detections = static_cast<int>(last_result.detections.size());
        session.ok = 1;
        return session;
    };

    for (int i = 0; i < warmup_runs; ++i) {
        BenchWire warmup = run_session(false, true);
        if (!warmup.ok) {
            fail(std::format("warmup {} failed: {}", i + 1, warmup.error.data()));
            return wire;
        }
    }

    std::vector<FrameTimingRow> measured_frame_timings;
    BenchWire measured = run_session(
        true, quiet, output_frame_timing_jsonl.empty() ? nullptr : &measured_frame_timings);
    if (!measured.ok) {
        wire = measured;
        return wire;
    }
    wire.t_frame0_ms = measured.t_frame0_ms;
    wire.t_frame0_encode_ms = measured.t_frame0_encode_ms;
    wire.t_frame0_prompt_ms = measured.t_frame0_prompt_ms;
    wire.t_frame0_segment_ms = measured.t_frame0_segment_ms;
    wire.t_frame0_tracker_add_ms = measured.t_frame0_tracker_add_ms;
    wire.t_frame0_tracker_mask_prepare_ms = measured.t_frame0_tracker_mask_prepare_ms;
    wire.t_frame0_tracker_memory_encode_ms = measured.t_frame0_tracker_memory_encode_ms;
    wire.t_frame0_tracker_obj_ptr_ms = measured.t_frame0_tracker_obj_ptr_ms;
    wire.t_frame0_tracker_store_ms = measured.t_frame0_tracker_store_ms;
    wire.t_frame0_tracker_unattributed_ms = measured.t_frame0_tracker_unattributed_ms;
    wire.t_frame0_prompt_unattributed_ms = measured.t_frame0_prompt_unattributed_ms;
    wire.t_track_avg_ms = measured.t_track_avg_ms;
    wire.t_track_p50_ms = measured.t_track_p50_ms;
    wire.t_track_p95_ms = measured.t_track_p95_ms;
    wire.t_total_ms = measured.t_total_ms;
    wire.t_session_ms = measured.t_session_ms;
    wire.t_model_e2e_ms = measured.t_model_e2e_ms;
    wire.t_required_e2e_ms = wire.t_input_load_ms + measured.t_model_e2e_ms;
    wire.t_cold_required_e2e_ms = wire.t_input_load_ms + wire.t_load_ms + measured.t_model_e2e_ms;
    wire.t_input_plus_session_ms = wire.t_input_load_ms + measured.t_session_ms;
    wire.t_required_accounted_ms = wire.t_input_load_ms + measured.t_model_accounted_ms;
    wire.t_required_accounting_delta_ms = wire.t_required_e2e_ms - wire.t_required_accounted_ms;
    wire.t_required_remainder_ms = std::max(0.0, wire.t_required_accounting_delta_ms);
    wire.t_model_accounting_delta_ms = measured.t_model_accounting_delta_ms;
    wire.t_model_accounted_ms = measured.t_model_accounted_ms;
    wire.t_session_overhead_ms = measured.t_session_overhead_ms;
    wire.t_model_remainder_ms = measured.t_model_remainder_ms;
    wire.t_state_create_ms = measured.t_state_create_ms;
    wire.t_tracker_create_ms = measured.t_tracker_create_ms;
    wire.t_session_setup_ms = measured.t_session_setup_ms;
    wire.t_artifact_ms = measured.t_artifact_ms;
    wire.t_frame0_artifact_ms = measured.t_frame0_artifact_ms;
    wire.t_frame0_model_ms = measured.t_frame0_model_ms;
    wire.t_frame0_prompt_model_ms = measured.t_frame0_prompt_model_ms;
    wire.t_track_all_ms = measured.t_track_all_ms;
    wire.t_track_all_model_ms = measured.t_track_all_model_ms;
    wire.t_tail_model_ms = measured.t_tail_model_ms;
    wire.t_tail_encode_required_total_ms = measured.t_tail_encode_required_total_ms;
    wire.t_required_tail_state_create_ms = measured.t_required_tail_state_create_ms;
    wire.t_required_tail_encode_ms = measured.t_required_tail_encode_ms;
    wire.t_required_tail_propagate_ms = measured.t_required_tail_propagate_ms;
    wire.t_required_tail_unattributed_ms = measured.t_required_tail_unattributed_ms;
    wire.t_required_output_artifact_ms = measured.t_required_output_artifact_ms;
    wire.t_tail_inline_encode_ms = measured.t_tail_inline_encode_ms;
    wire.t_tail_preencode_encode_ms = measured.t_tail_preencode_encode_ms;
    wire.t_tail_inline_propagate_ms = measured.t_tail_inline_propagate_ms;
    wire.t_tail_preencoded_propagate_ms = measured.t_tail_preencoded_propagate_ms;
    wire.t_frame0_encode_graph_compute_ms = measured.t_frame0_encode_graph_compute_ms;
    wire.t_tail_inline_encode_graph_compute_ms = measured.t_tail_inline_encode_graph_compute_ms;
    wire.t_tail_preencode_encode_graph_compute_ms =
        measured.t_tail_preencode_encode_graph_compute_ms;
    wire.t_tail_encode_graph_compute_ms = measured.t_tail_encode_graph_compute_ms;
    wire.t_tail_propagate_graph_compute_ms = measured.t_tail_propagate_graph_compute_ms;
    wire.t_tail_propagate_cache_compute_ms = measured.t_tail_propagate_cache_compute_ms;
    wire.t_tail_propagate_input_prompt_upload_ms = measured.t_tail_propagate_input_prompt_upload_ms;
    wire.t_tail_propagate_input_rope_upload_ms = measured.t_tail_propagate_input_rope_upload_ms;
    wire.t_tail_propagate_input_memory_upload_ms = measured.t_tail_propagate_input_memory_upload_ms;
    wire.t_tail_propagate_input_feature_upload_ms =
        measured.t_tail_propagate_input_feature_upload_ms;
    wire.t_tail_propagate_input_constant_upload_ms =
        measured.t_tail_propagate_input_constant_upload_ms;
    wire.t_tail_propagate_input_sparse_upload_ms = measured.t_tail_propagate_input_sparse_upload_ms;
    wire.t_required_core_compute_ms = measured.t_required_core_compute_ms;
    wire.t_required_non_core_compute_ms =
        std::max(0.0, wire.t_required_e2e_ms - wire.t_required_core_compute_ms);
    wire.t_required_core_compute_fraction =
        wire.t_required_e2e_ms > 0.0 ? wire.t_required_core_compute_ms / wire.t_required_e2e_ms
                                     : 0.0;
    wire.t_tail_encode_graph_compute_fraction = measured.t_tail_encode_graph_compute_fraction;
    wire.t_track_encode_avg_ms = measured.t_track_encode_avg_ms;
    wire.t_track_encode_total_ms = measured.t_track_encode_total_ms;
    wire.t_track_propagate_avg_ms = measured.t_track_propagate_avg_ms;
    wire.t_track_propagate_total_ms = measured.t_track_propagate_total_ms;
    wire.t_track_unattributed_avg_ms = measured.t_track_unattributed_avg_ms;
    wire.t_track_unattributed_total_ms = measured.t_track_unattributed_total_ms;
    wire.t_track_artifact_avg_ms = measured.t_track_artifact_avg_ms;
    wire.t_track_artifact_total_ms = measured.t_track_artifact_total_ms;
    wire.t_preencode_avg_ms = measured.t_preencode_avg_ms;
    wire.t_preencode_total_ms = measured.t_preencode_total_ms;
    wire.t_preencode_state_create_avg_ms = measured.t_preencode_state_create_avg_ms;
    wire.t_preencode_state_create_total_ms = measured.t_preencode_state_create_total_ms;
    wire.frame0_encode_timing = measured.frame0_encode_timing;
    wire.n_track_frames = measured.n_track_frames;
    wire.n_track_all_frames = measured.n_track_all_frames;
    wire.n_track_split_frames = measured.n_track_split_frames;
    wire.n_preencoded_frames = measured.n_preencoded_frames;
    wire.n_frame0_candidates = measured.n_frame0_candidates;
    wire.n_frame0_added_instances = measured.n_frame0_added_instances;
    wire.text_init_selected_only = text_init_selected_only ? 1 : 0;
    wire.n_detections = measured.n_detections;
    if (!write_frame_timing_rows(output_frame_timing_jsonl,
                                 model_path,
                                 wire.backend.data(),
                                 wire,
                                 measured_frame_timings)) {
        fail("write frame timing JSONL failed");
        return wire;
    }
#ifndef _WIN32
    {
        struct rusage usage = {};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            wire.max_rss_kib = max_rss_to_kib(usage.ru_maxrss);
        }
    }
#endif
    wire.ok = 1;

    return wire;
}

// ── Isolated benchmark execution ───────────────────────────────────────────

#ifndef _WIN32
// POSIX: fork a subprocess for crash isolation
static void child_benchmark(const std::string& model_path,
                            bool use_gpu,
                            const std::string& video_path,
                            const std::string& frame_dir,
                            int n_frames,
                            float px,
                            float py,
                            const std::string& text_prompt,
                            int n_threads,
                            int encode_img_size,
                            bool bbox_only,
                            bool multimask,
                            std::optional<size_t> initial_candidate_index,
                            int recondition_every,
                            const std::string& output_jsonl,
                            const std::string& output_initial_candidates_jsonl,
                            const std::string& output_frame_timing_jsonl,
                            const std::string& output_mask_dir,
                            const std::string& output_logits_dir,
                            int warmup_runs,
                            bool preencode_track_frames,
                            bool preencode_cached_tail_frames,
                            bool text_init_selected_only,
                            int timed_start_frame,
                            bool quiet,
                            int write_fd) {
    BenchWire wire = run_single_benchmark(model_path,
                                          use_gpu,
                                          video_path,
                                          frame_dir,
                                          n_frames,
                                          px,
                                          py,
                                          text_prompt,
                                          n_threads,
                                          encode_img_size,
                                          bbox_only,
                                          multimask,
                                          initial_candidate_index,
                                          recondition_every,
                                          output_jsonl,
                                          output_initial_candidates_jsonl,
                                          output_frame_timing_jsonl,
                                          output_mask_dir,
                                          output_logits_dir,
                                          warmup_runs,
                                          preencode_track_frames,
                                          preencode_cached_tail_frames,
                                          text_init_selected_only,
                                          timed_start_frame,
                                          quiet);
    const auto bytes = std::as_bytes(std::span{&wire, 1});
    if (!write_full(write_fd, bytes)) {
        std::print(stderr, "child_benchmark: failed to write result pipe\n");
    }
    close(write_fd);
    _exit(wire.ok ? 0 : 1);
}
#endif

static BenchResult run_benchmark_isolated(
    const ModelEntry& entry,
    bool use_gpu,
    const std::string& video_path,
    const std::string& frame_dir,
    int n_frames,
    float px,
    float py,
    const std::string& text_prompt,
    int n_threads,
    int encode_img_size = 0,
    bool bbox_only = false,
    bool multimask = false,
    std::optional<size_t> initial_candidate_index = std::nullopt,
    int recondition_every = 16,
    const std::string& output_jsonl = "",
    const std::string& output_initial_candidates_jsonl = "",
    const std::string& output_frame_timing_jsonl = "",
    const std::string& output_mask_dir = "",
    const std::string& output_logits_dir = "",
    int warmup_runs = 0,
    bool preencode_track_frames = false,
    bool preencode_cached_tail_frames = false,
    bool text_init_selected_only = false,
    int timed_start_frame = 1,
    bool quiet = false) {
    BenchResult res;
    res.model_name = entry.name;
    res.backend = requested_backend_label(use_gpu);
    res.file_size = entry.file_size;

#ifdef _WIN32
    // Windows: run in-process (no crash isolation)
    BenchWire wire = run_single_benchmark(entry.path,
                                          use_gpu,
                                          video_path,
                                          frame_dir,
                                          n_frames,
                                          px,
                                          py,
                                          text_prompt,
                                          n_threads,
                                          encode_img_size,
                                          bbox_only,
                                          multimask,
                                          initial_candidate_index,
                                          recondition_every,
                                          output_jsonl,
                                          output_initial_candidates_jsonl,
                                          output_frame_timing_jsonl,
                                          output_mask_dir,
                                          output_logits_dir,
                                          warmup_runs,
                                          preencode_track_frames,
                                          preencode_cached_tail_frames,
                                          text_init_selected_only,
                                          timed_start_frame,
                                          quiet);
    if (wire.ok) {
        apply_successful_wire(res, wire, use_gpu);
    } else {
        res.error = wire.error.data();
    }
#else
    auto pipe_result = make_pipe();
    if (!pipe_result) {
        res.error = pipe_result.error();
        return res;
    }
    Pipe pipe = std::move(*pipe_result);

    pid_t pid = fork();
    if (pid < 0) {
        pipe.read.reset();
        pipe.write.reset();
        res.error = "fork() failed";
        return res;
    }

    if (pid == 0) {
        // Child
        pipe.read.reset();
        const int write_fd = pipe.write.release();
        child_benchmark(entry.path,
                        use_gpu,
                        video_path,
                        frame_dir,
                        n_frames,
                        px,
                        py,
                        text_prompt,
                        n_threads,
                        encode_img_size,
                        bbox_only,
                        multimask,
                        initial_candidate_index,
                        recondition_every,
                        output_jsonl,
                        output_initial_candidates_jsonl,
                        output_frame_timing_jsonl,
                        output_mask_dir,
                        output_logits_dir,
                        warmup_runs,
                        preencode_track_frames,
                        preencode_cached_tail_frames,
                        text_init_selected_only,
                        timed_start_frame,
                        quiet,
                        write_fd);
        _exit(1);
    }

    // Parent
    pipe.write.reset();

    BenchWire wire = {};
    bool got_wire = read_full(pipe.read.get(), std::as_writable_bytes(std::span{&wire, 1}));
    pipe.read.reset();

    int status = 0;
    waitpid(pid, &status, 0);

    if (got_wire && wire.ok) {
        apply_successful_wire(res, wire, use_gpu);
    } else if (got_wire && !wire.ok) {
        res.error = wire.error.data();
    } else if (WIFSIGNALED(status)) {
        res.error = std::format("crashed (signal {})", WTERMSIG(status));
    } else {
        res.error = "child failed (no result)";
    }
#endif

    return res;
}

static BenchResult run_benchmark_direct(
    const ModelEntry& entry,
    bool use_gpu,
    const std::string& video_path,
    const std::string& frame_dir,
    int n_frames,
    float px,
    float py,
    const std::string& text_prompt,
    int n_threads,
    int encode_img_size = 0,
    bool bbox_only = false,
    bool multimask = false,
    std::optional<size_t> initial_candidate_index = std::nullopt,
    int recondition_every = 16,
    const std::string& output_jsonl = "",
    const std::string& output_initial_candidates_jsonl = "",
    const std::string& output_frame_timing_jsonl = "",
    const std::string& output_mask_dir = "",
    const std::string& output_logits_dir = "",
    int warmup_runs = 0,
    bool preencode_track_frames = false,
    bool preencode_cached_tail_frames = false,
    bool text_init_selected_only = false,
    int timed_start_frame = 1,
    bool quiet = false) {
    BenchResult res;
    res.model_name = entry.name;
    res.backend = requested_backend_label(use_gpu);
    res.file_size = entry.file_size;

    BenchWire wire = run_single_benchmark(entry.path,
                                          use_gpu,
                                          video_path,
                                          frame_dir,
                                          n_frames,
                                          px,
                                          py,
                                          text_prompt,
                                          n_threads,
                                          encode_img_size,
                                          bbox_only,
                                          multimask,
                                          initial_candidate_index,
                                          recondition_every,
                                          output_jsonl,
                                          output_initial_candidates_jsonl,
                                          output_frame_timing_jsonl,
                                          output_mask_dir,
                                          output_logits_dir,
                                          warmup_runs,
                                          preencode_track_frames,
                                          preencode_cached_tail_frames,
                                          text_init_selected_only,
                                          timed_start_frame,
                                          quiet);
    if (wire.ok) {
        apply_successful_wire(res, wire, use_gpu);
    } else {
        res.error = wire.error.data();
    }
    return res;
}

// ── Table printing ──────────────────────────────────────────────────────────

static void print_table(const std::vector<BenchResult>& results,
                        const std::string& video_path,
                        float px,
                        float py,
                        int n_frames,
                        int n_threads) {
    std::print("\n");
    std::print(
        "=========================================================================================="
        "===============\n");
    std::print("SAM3.CPP BENCHMARK  —  {} frames, point=({:.1f}, {:.1f}), threads={}\n",
               n_frames,
               px,
               py,
               n_threads);
    std::print("video: {}\n", video_path);
    std::print(
        "=========================================================================================="
        "===============\n\n");

    std::print(
        "  {:>3} | {:<36} | {:>7} | {:>7} | {:>9} | {:>9} | {:>13} | {:>9} | {:>9} | {:>10} | "
        "{:>8} | {:>3} | {}\n",
        "#",
        "Model",
        "Size",
        "Backend",
        "Load (ms)",
        "Init (ms)",
        "Track/fr (ms)",
        "P50 (ms)",
        "P95 (ms)",
        "Total (ms)",
        "RSS MiB",
        "Det",
        "Status");
    std::print(
        "------+--------------------------------------+---------+---------+-----------+-----------+"
        "---------------+-----------+-----------+------------+----------+-----+--------\n");

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        if (r.success) {
            std::print(
                "  {:>3} | {:<36} | {:>7} | {:>7} | {:>9} | {:>9} | {:>13} | {:>9} | {:>9} | "
                "{:>10} | {:>8.1f} | {:>3} | OK\n",
                i + 1,
                r.model_name,
                format_size(r.file_size),
                r.backend,
                format_time_short(r.t_load_ms),
                format_time_short(r.t_frame0_ms),
                format_time_short(r.t_track_avg_ms),
                format_time_short(r.t_track_p50_ms),
                format_time_short(r.t_track_p95_ms),
                format_time_short(r.t_total_ms),
                r.max_rss_kib / 1024.0,
                r.n_detections);
        } else {
            std::print(
                "  {:>3} | {:<36} | {:>7} | {:>7} | {:>9} | {:>9} | {:>13} | {:>9} | {:>9} | "
                "{:>10} | {:>8} | {:>3} | FAIL: {}\n",
                i + 1,
                r.model_name,
                format_size(r.file_size),
                r.backend,
                r.t_load_ms > 0 ? format_time_short(r.t_load_ms) : "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                r.error);
        }
    }

    std::print(
        "------+--------------------------------------+---------+---------+-----------+-----------+"
        "---------------+-----------+-----------+------------+----------+-----+--------\n");

    int n_ok = 0, n_fail = 0;
    for (const auto& r : results) {
        if (r.success)
            n_ok++;
        else
            n_fail++;
    }
    std::print("\nSUMMARY: {} runs, {} OK, {} FAIL\n", results.size(), n_ok, n_fail);
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string models_dir = "models/";
    std::string video_path = "data/test_video.mp4";
    std::string frame_dir;
    float px = 315.0f;
    float py = 250.0f;
    std::string text_prompt = "person";
    int n_frames = 10;
    int n_threads = 4;
    int encode_img_size = 0;
    int recondition_every = 16;
    int warmup_runs = 0;
    int timed_start_frame = 1;
    bool cpu_only = false;
    bool gpu_only = false;
    bool bbox_only = false;
    bool multimask = false;
    bool no_isolation = false;
    bool preencode_track_frames = false;
    bool preencode_cached_tail_frames = false;
    bool text_init_selected_only = false;
    bool no_output_artifacts = false;
    bool quiet = false;
    std::optional<size_t> initial_candidate_index;
    std::string filter;
    std::string output_jsonl;
    std::string output_initial_candidates_jsonl;
    std::string output_frame_timing_jsonl;
    std::string output_mask_dir;
    std::string output_logits_dir;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--frame-dir" && i + 1 < argc) {
            frame_dir = argv[++i];
        } else if (arg == "--point-x" && i + 1 < argc) {
            px = (float) atof(argv[++i]);
        } else if (arg == "--point-y" && i + 1 < argc) {
            py = (float) atof(argv[++i]);
        } else if (arg == "--text-prompt" && i + 1 < argc) {
            text_prompt = argv[++i];
        } else if (arg == "--n-frames" && i + 1 < argc) {
            n_frames = atoi(argv[++i]);
        } else if (arg == "--n-threads" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (arg == "--encode-img-size" && i + 1 < argc) {
            encode_img_size = atoi(argv[++i]);
        } else if (arg == "--recondition-every" && i + 1 < argc) {
            recondition_every = atoi(argv[++i]);
        } else if (arg == "--warmup-runs" && i + 1 < argc) {
            warmup_runs = atoi(argv[++i]);
        } else if (arg == "--timed-start-frame" && i + 1 < argc) {
            timed_start_frame = atoi(argv[++i]);
        } else if (arg == "--cpu-only") {
            cpu_only = true;
        } else if (arg == "--gpu-only") {
            gpu_only = true;
        } else if (arg == "--bbox-only") {
            bbox_only = true;
        } else if (arg == "--multimask") {
            multimask = true;
        } else if (arg == "--preencode-track-frames") {
            preencode_track_frames = true;
        } else if (arg == "--preencode-cached-tail-frames") {
            preencode_cached_tail_frames = true;
        } else if (arg == "--text-init-selected-only") {
            text_init_selected_only = true;
        } else if (arg == "--no-output-artifacts") {
            no_output_artifacts = true;
        } else if (arg == "--initial-candidate-index" && i + 1 < argc) {
            initial_candidate_index = parse_size_t(argv[++i]);
            if (!initial_candidate_index) {
                std::print(stderr, "ERROR: invalid --initial-candidate-index value\n");
                return 1;
            }
        } else if (arg == "--no-isolation") {
            no_isolation = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--filter" && i + 1 < argc) {
            filter = argv[++i];
        } else if (arg == "--output-jsonl" && i + 1 < argc) {
            output_jsonl = argv[++i];
        } else if (arg == "--output-initial-candidates-jsonl" && i + 1 < argc) {
            output_initial_candidates_jsonl = argv[++i];
        } else if (arg == "--output-frame-timing-jsonl" && i + 1 < argc) {
            output_frame_timing_jsonl = argv[++i];
        } else if (arg == "--output-mask-dir" && i + 1 < argc) {
            output_mask_dir = argv[++i];
        } else if (arg == "--output-logits-dir" && i + 1 < argc) {
            output_logits_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::print(
                stderr,
                "Usage: {} [options]\n"
                "  --models-dir <path>   Models directory       (default: models/)\n"
                "  --video <path>        Video file             (default: data/test_video.mp4)\n"
                "  --frame-dir <path>    Directory of extracted frames, used instead of --video\n"
                "  --point-x <f>         Click X                (default: 315.0)\n"
                "  --point-y <f>         Click Y                (default: 250.0)\n"
                "  --text-prompt <text>  SAM3 text prompt       (default: person)\n"
                "  --n-frames <n>        Frames to track        (default: 10)\n"
                "  --n-threads <n>       CPU threads            (default: 4)\n"
                "  --recondition-every <n> Memory refresh interval (default: 16)\n"
                "  --warmup-runs <n>     Run untimed same-model sessions before measurement\n"
                "  --timed-start-frame <n> Count Track/fr timing from this frame index (default: "
                "1)\n"
                "  --cpu-only            Skip GPU runs\n"
                "  --gpu-only            Skip CPU runs\n"
                "  --bbox-only           Track/output bbox rows without full-res masks\n"
                "  --multimask           Use multimask output for the initial point prompt\n"
                "  --preencode-track-frames Pre-encode frames 1..N-1 and time propagation only\n"
                "  --preencode-cached-tail-frames Pre-encode frames 2..N-1; frame 1 keeps normal "
                "encode+propagate timing\n"
                "  --text-init-selected-only Track only the selected SAM3 text-init detection\n"
                "  --no-output-artifacts Skip benchmark JSONL/mask/logits artifact generation in "
                "the measured run\n"
                "  --initial-candidate-index <n> Force a point-prompt candidate for diagnostics\n"
                "  --filter <substr>     Filter model filenames\n"
                "  --output-jsonl <path> Write first-run target bbox rows\n"
                "  --output-initial-candidates-jsonl <path> Write initial point-prompt candidates\n"
                "  --output-frame-timing-jsonl <path> Write per-frame timing rows after "
                "measurement\n"
                "  --output-mask-dir <path> Write first-run target masks as PNG files\n"
                "  --output-logits-dir <path> Write first-run selected low-res logits as "
                ".bin/.shape\n"
                "  --no-isolation        Run in-process for profiler capture\n"
                "  --quiet               Suppress per-frame progress lines\n",
                argv[0]);
            return 0;
        } else {
            std::print(stderr, "Unknown argument: {}\n", arg);
            return 1;
        }
    }

    if (quiet && std::getenv("SAM3_LOG_RUNTIME_LEVEL") == nullptr) {
#ifdef _WIN32
        _putenv_s("SAM3_LOG_RUNTIME_LEVEL", "0");
#else
        setenv("SAM3_LOG_RUNTIME_LEVEL", "0", 0);
#endif
    }

    if ((!output_jsonl.empty() || !output_initial_candidates_jsonl.empty() ||
         !output_frame_timing_jsonl.empty() || !output_mask_dir.empty() ||
         !output_logits_dir.empty()) &&
        !(gpu_only || cpu_only)) {
        std::print(stderr,
                   "ERROR: output JSONL/mask/timing options require --gpu-only or --cpu-only to "
                   "select one "
                   "backend\n");
        return 1;
    }

    if (n_frames < 2) {
        std::print(stderr, "ERROR: --n-frames must be >= 2\n");
        return 1;
    }
    if (recondition_every < 1) {
        std::print(stderr, "ERROR: --recondition-every must be >= 1\n");
        return 1;
    }
    if (warmup_runs < 0) {
        std::print(stderr, "ERROR: --warmup-runs must be >= 0\n");
        return 1;
    }
    if (timed_start_frame < 1 || timed_start_frame >= n_frames) {
        std::print(stderr, "ERROR: --timed-start-frame must satisfy 1 <= value < --n-frames\n");
        return 1;
    }
    if (preencode_track_frames && preencode_cached_tail_frames) {
        std::print(stderr,
                   "ERROR: --preencode-track-frames and --preencode-cached-tail-frames are "
                   "mutually exclusive\n");
        return 1;
    }

    // Discover models
    auto entries = discover_models(models_dir, filter);
    if (entries.empty()) {
        std::print(stderr, "ERROR: no .ggml files found in '{}'", models_dir);
        if (!filter.empty()) {
            std::print(stderr, " (filter: '{}')", filter);
        }
        std::print(stderr, "\n");
        return 1;
    }

    std::print(stderr, "Found {} model(s)\n", entries.size());
    for (const auto& e : entries) {
        std::print(stderr, "  {}  ({})\n", e.name, format_size(e.file_size));
    }

    if (frame_dir.empty()) {
        auto vinfo = sam3_get_video_info(video_path);
        if (vinfo.n_frames <= 0) {
            std::print(stderr, "ERROR: cannot read video '{}'\n", video_path);
            return 1;
        }
        if (n_frames > vinfo.n_frames) {
            std::print(stderr,
                       "WARNING: video has {} frames, clamping to {}\n",
                       vinfo.n_frames,
                       vinfo.n_frames);
            n_frames = vinfo.n_frames;
        }
        std::print(stderr,
                   "Video: {}x{}, {} frames, {:.1f} fps\n",
                   vinfo.width,
                   vinfo.height,
                   vinfo.n_frames,
                   vinfo.fps);
    } else {
        std::error_code ec;
        if (!std::filesystem::is_directory(frame_dir, ec)) {
            std::print(stderr, "ERROR: cannot read frame directory '{}'\n", frame_dir);
            return 1;
        }
        std::print(stderr, "Frames: {} ({} requested)\n", frame_dir, n_frames);
    }

    // Build run list
    struct RunSpec {
        const ModelEntry* entry;
        bool use_gpu;
    };
    std::vector<RunSpec> runs;
    for (const auto& e : entries) {
        if (!gpu_only)
            runs.push_back({&e, false});
        if (!cpu_only)
            runs.push_back({&e, true});
    }

    std::sort(runs.begin(), runs.end(), [](const RunSpec& a, const RunSpec& b) {
        int ka = model_sort_key(a.entry->name);
        int kb = model_sort_key(b.entry->name);
        if (ka != kb)
            return ka < kb;
        return a.use_gpu > b.use_gpu;  // GPU first
    });

    if (no_isolation) {
        std::print(stderr, "\nStarting {} benchmark runs (in-process)...\n\n", runs.size());
    } else {
#ifdef _WIN32
        std::print(stderr, "\nStarting {} benchmark runs (in-process)...\n\n", runs.size());
#else
        std::print(
            stderr, "\nStarting {} benchmark runs (each in a subprocess)...\n\n", runs.size());
#endif
    }

    // Run benchmarks
    int64_t t_wall_start = ggml_time_us();
    std::vector<BenchResult> results;
    results.reserve(runs.size());

    for (size_t i = 0; i < runs.size(); i++) {
        const auto& run = runs[i];
        const char* backend_str = requested_backend_label(run.use_gpu);

        std::print(
            stderr, "[{:>3}/{}] {} ({}) ...\n", i + 1, runs.size(), run.entry->name, backend_str);

        const std::string measured_output_jsonl =
            no_output_artifacts ? "" : ((i == 0) ? output_jsonl : "");
        const std::string measured_output_initial_candidates_jsonl =
            no_output_artifacts ? "" : ((i == 0) ? output_initial_candidates_jsonl : "");
        const std::string measured_output_frame_timing_jsonl =
            (i == 0) ? output_frame_timing_jsonl : "";
        const std::string measured_output_mask_dir =
            no_output_artifacts ? "" : ((i == 0) ? output_mask_dir : "");
        const std::string measured_output_logits_dir =
            no_output_artifacts ? "" : ((i == 0) ? output_logits_dir : "");

        auto res = no_isolation ? run_benchmark_direct(*run.entry,
                                                       run.use_gpu,
                                                       video_path,
                                                       frame_dir,
                                                       n_frames,
                                                       px,
                                                       py,
                                                       text_prompt,
                                                       n_threads,
                                                       encode_img_size,
                                                       bbox_only,
                                                       multimask,
                                                       initial_candidate_index,
                                                       recondition_every,
                                                       measured_output_jsonl,
                                                       measured_output_initial_candidates_jsonl,
                                                       measured_output_frame_timing_jsonl,
                                                       measured_output_mask_dir,
                                                       measured_output_logits_dir,
                                                       warmup_runs,
                                                       preencode_track_frames,
                                                       preencode_cached_tail_frames,
                                                       text_init_selected_only,
                                                       timed_start_frame,
                                                       quiet)
                                : run_benchmark_isolated(*run.entry,
                                                         run.use_gpu,
                                                         video_path,
                                                         frame_dir,
                                                         n_frames,
                                                         px,
                                                         py,
                                                         text_prompt,
                                                         n_threads,
                                                         encode_img_size,
                                                         bbox_only,
                                                         multimask,
                                                         initial_candidate_index,
                                                         recondition_every,
                                                         measured_output_jsonl,
                                                         measured_output_initial_candidates_jsonl,
                                                         measured_output_frame_timing_jsonl,
                                                         measured_output_mask_dir,
                                                         measured_output_logits_dir,
                                                         warmup_runs,
                                                         preencode_track_frames,
                                                         preencode_cached_tail_frames,
                                                         text_init_selected_only,
                                                         timed_start_frame,
                                                         quiet);
        results.push_back(res);
        print_result_json(i + 1, res);

        if (res.success) {
            std::print(stderr,
                       "  -> OK  backend={}  load={:.0f}ms  init={:.0f}ms  track/fr={:.0f}ms  "
                       "p50={:.0f}ms  "
                       "p95={:.0f}ms  total={:.0f}ms  rss={:.1f}MiB  det={}\n\n",
                       res.backend,
                       res.t_load_ms,
                       res.t_frame0_ms,
                       res.t_track_avg_ms,
                       res.t_track_p50_ms,
                       res.t_track_p95_ms,
                       res.t_total_ms,
                       res.max_rss_kib / 1024.0,
                       res.n_detections);
        } else {
            std::print(stderr, "  -> FAIL: {}\n\n", res.error);
        }
    }

    double t_wall_ms = (ggml_time_us() - t_wall_start) / 1000.0;

    // Print table
    print_table(results, video_path, px, py, n_frames, n_threads);

    double t_wall_s = t_wall_ms / 1000.0;
    int mins = (int) (t_wall_s / 60.0);
    int secs = (int) (t_wall_s) % 60;
    std::print("Total wall time: {}m {}s\n\n", mins, secs);

    return 0;
}
