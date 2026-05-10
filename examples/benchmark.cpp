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
 *   --point-x <f>         Click point X                (default: 315.0)
 *   --point-y <f>         Click point Y                (default: 250.0)
 *   --n-frames <n>        Frames to track              (default: 10)
 *   --n-threads <n>       CPU threads                  (default: 4)
 *   --recondition-every <n> Memory refresh interval    (default: 16)
 *   --cpu-only            Skip GPU runs
 *   --gpu-only            Skip CPU runs
 *   --bbox-only           Track/output bbox rows without full-res masks
 *   --multimask           Use multimask output for the initial point prompt
 *   --filter <substr>     Only run models whose filename contains <substr>
 *   --output-jsonl <path> Write first-run target bbox rows for quality checks
 *   --output-initial-candidates-jsonl <path> Write initial point-prompt candidates
 *   --output-mask-dir <path> Write first-run target masks as PNG files
 *   --no-isolation        Run in-process for profilers that do not follow fork
 */

#include "sam3.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <format>
#include <ostream>
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
    double t_track_avg_ms;
    double t_track_p50_ms;
    double t_track_p95_ms;
    double t_total_ms;
    int n_track_frames;
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
    double t_track_avg_ms = 0.0;
    double t_track_p50_ms = 0.0;
    double t_track_p95_ms = 0.0;
    double t_total_ms = 0.0;
    long max_rss_kib = 0;
    int n_detections = 0;
    bool success = false;
    std::string error;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string format_size(int64_t bytes) {
    char buf[32];
    if (bytes >= (int64_t) 1024 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= (int64_t) 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%lld MB", (long long) (bytes / (1024 * 1024)));
    } else {
        snprintf(buf, sizeof(buf), "%lld KB", (long long) (bytes / 1024));
    }
    return buf;
}

static std::string format_time_short(double ms) {
    if (ms < 0)
        return "  -";
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", ms);
    return buf;
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

class FileHandle {
public:
    FileHandle() = default;

    explicit FileHandle(FILE* file) : file_(file) {}

    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FileHandle(FileHandle&& other) noexcept : file_(std::exchange(other.file_, nullptr)) {}

    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            reset();
            file_ = std::exchange(other.file_, nullptr);
        }
        return *this;
    }

    ~FileHandle() { reset(); }

    [[nodiscard]] FILE* get() const noexcept { return file_; }

    [[nodiscard]] explicit operator bool() const noexcept { return file_ != nullptr; }

    void reset(FILE* file = nullptr) noexcept {
        if (file_) {
            fclose(file_);
        }
        file_ = file;
    }

    [[nodiscard]] static std::expected<FileHandle, std::string> open_write(std::string_view path) {
        std::string owned_path{path};
        FILE* file = fopen(owned_path.c_str(), "w");
        if (!file) {
            return std::unexpected("open output_jsonl failed");
        }
        return FileHandle{file};
    }

private:
    FILE* file_ = nullptr;
};

static bool ends_with(std::string_view s, std::string_view suffix) {
    if (suffix.size() > s.size())
        return false;
    return s.substr(s.size() - suffix.size()) == suffix;
}

static std::string strip_extension(std::string_view filename) {
    auto pos = filename.rfind('.');
    return std::string{(pos != std::string_view::npos) ? filename.substr(0, pos) : filename};
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

static std::string display_backend_name(std::string_view backend_name, bool requested_gpu) {
    if (!requested_gpu)
        return "CPU";
    if (backend_name.starts_with("CUDA"))
        return "CUDA";
    if (backend_name.starts_with("Metal"))
        return "Metal";
    return backend_name.empty() ? requested_backend_label(true) : std::string{backend_name};
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

static const sam3_detection* best_detection(const sam3_result& result) {
    if (result.detections.empty())
        return nullptr;
    return &*std::ranges::max_element(
        result.detections, {}, [](const sam3_detection& det) { return det.iou_score; });
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

static void write_metadata_row(FILE* out,
                               const std::string& model_path,
                               std::string_view backend,
                               const std::string& video_path,
                               const sam3_image& first_frame,
                               int n_frames,
                               float px,
                               float py,
                               int requested_encode_img_size,
                               int effective_encode_img_size,
                               bool bbox_only,
                               bool multimask) {
    if (!out)
        return;
    const std::string model_path_json = json_escape(model_path);
    const std::string backend_json = json_escape(backend);
    const std::string video_path_json = json_escape(video_path);
    fprintf(out,
            "{\"source\":\"sam3cpp-meta\","
            "\"model_path\":\"%s\","
            "\"backend\":\"%s\","
            "\"video_path\":\"%s\","
            "\"decoded_width\":%d,\"decoded_height\":%d,"
            "\"frames\":%d,"
            "\"point_x\":%.6f,\"point_y\":%.6f,"
            "\"encode_img_size_requested\":%d,"
            "\"encode_img_size_effective\":%d,"
            "\"bbox_only\":%s,\"multimask\":%s}\n",
            model_path_json.c_str(),
            backend_json.c_str(),
            video_path_json.c_str(),
            first_frame.width,
            first_frame.height,
            n_frames,
            px,
            py,
            requested_encode_img_size,
            effective_encode_img_size,
            bbox_only ? "true" : "false",
            multimask ? "true" : "false");
}

static void write_detection_row(FILE* out,
                                int offset,
                                int expected_frame_index,
                                const sam3_result& result,
                                std::string_view mask_path = {}) {
    if (!out)
        return;
    if (result.detections.empty()) {
        fprintf(out,
                "{\"offset\":%d,\"expected_frame_index\":%d,\"bbox_xyxy\":null,"
                "\"score\":0,\"mask_area\":0,\"mask_fnv1a64\":\"0000000000000000\","
                "\"mask_path\":null,"
                "\"source\":\"sam3cpp-missing\"}\n",
                offset,
                expected_frame_index);
        return;
    }
    const auto* best = best_detection(result);
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
    fprintf(out,
            "{\"offset\":%d,\"expected_frame_index\":%d,"
            "\"bbox_xyxy\":[%.3f,%.3f,%.3f,%.3f],\"score\":%.6f,"
            "\"mask_area\":%d,\"mask_fnv1a64\":\"%016llx\","
            "\"mask_path\":%s%s%s,"
            "\"source\":\"sam3cpp-edgetam\"}\n",
            offset,
            expected_frame_index,
            det.box.x0,
            det.box.y0,
            det.box.x1,
            det.box.y1,
            det.score,
            mask_area,
            (unsigned long long) mask_hash,
            mask_path.empty() ? "" : "\"",
            mask_path.empty() ? "null" : std::string{mask_path}.c_str(),
            mask_path.empty() ? "" : "\"");
}

static void write_initial_candidate_rows(std::ostream* out, const sam3_result& result) {
    if (!out)
        return;
    for (size_t candidate_index = 0; candidate_index < result.detections.size(); ++candidate_index) {
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
            "\"instance_id\":{},"
            "\"bbox_xyxy\":[{:.3f},{:.3f},{:.3f},{:.3f}],"
            "\"score\":{:.6f},"
            "\"iou_score\":{:.6f},"
            "\"obj_score\":{:.6f},"
            "\"mask_area\":{},"
            "\"mask_fnv1a64\":\"{:016x}\"}}\n",
            candidate_index,
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
                                       const sam3_result& result) {
    const auto* det = best_detection(result);
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
        fprintf(stderr, "ERROR: cannot open models directory '%s'\n", dir.c_str());
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
        fprintf(stderr, "ERROR: cannot open models directory '%s'\n", dir.c_str());
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
                                      int n_frames,
                                      float px,
                                      float py,
                                      int n_threads,
                                      int encode_img_size,
                                      bool bbox_only,
                                      bool multimask,
                                      int recondition_every,
                                      const std::string& output_jsonl,
                                      const std::string& output_initial_candidates_jsonl,
                                      const std::string& output_mask_dir,
                                      bool quiet) {
    BenchWire wire = {};

    auto fail = [&](const char* msg) {
        wire.ok = 0;
        snprintf(wire.error.data(), wire.error.size(), "%s", msg);
    };

    FileHandle out;
    if (!output_jsonl.empty()) {
        auto opened = FileHandle::open_write(output_jsonl);
        if (!opened) {
            fail(opened.error().c_str());
            return wire;
        }
        out = std::move(*opened);
    }

    std::ofstream candidate_out;
    if (!output_initial_candidates_jsonl.empty()) {
        candidate_out.open(output_initial_candidates_jsonl);
        if (!candidate_out) {
            fail("open initial candidates JSONL failed");
            return wire;
        }
    }

    // Decode frames
    std::vector<sam3_image> frames(n_frames);
    for (int f = 0; f < n_frames; f++) {
        frames[f] = sam3_decode_video_frame(video_path, f);
        if (frames[f].data.empty()) {
            fail("decode frame failed");
            return wire;
        }
    }

    // Load model
    int64_t t0 = ggml_time_us();

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = use_gpu;
    params.n_threads = n_threads;
    params.encode_img_size = encode_img_size;

    auto model = sam3_load_model(params);
    if (!model) {
        fail("load failed");
        return wire;
    }
    snprintf(wire.backend.data(), wire.backend.size(), "%s", sam3_backend_name(*model));
    if (out) {
        const int effective_encode_img_size =
            (encode_img_size > 0) ? encode_img_size : sam3_model_image_size(*model);
        write_metadata_row(out.get(),
                           model_path,
                           sam3_backend_name(*model),
                           video_path,
                           frames[0],
                           n_frames,
                           px,
                           py,
                           encode_img_size,
                           effective_encode_img_size,
                           bbox_only,
                           multimask);
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fail("state failed");
        return wire;
    }

    bool visual_only = sam3_is_visual_only(*model);
    sam3_tracker_ptr tracker;

    if (visual_only) {
        sam3_visual_track_params vtp;
        vtp.max_keep_alive = 100;
        vtp.recondition_every = recondition_every;
        vtp.bbox_only = bbox_only;
        tracker = sam3_create_visual_tracker(*model, vtp);
    } else {
        sam3_video_params vp;
        vp.hotstart_delay = 0;
        vp.max_keep_alive = 100;
        vp.recondition_every = recondition_every;
        tracker = sam3_create_tracker(*model, vp);
    }
    if (!tracker) {
        fail("tracker failed");
        return wire;
    }

    wire.t_load_ms = (ggml_time_us() - t0) / 1000.0;

    // Frame 0: encode + add instance
    t0 = ggml_time_us();

    if (!sam3_encode_image(*state, *model, frames[0])) {
        fail("encode f0 failed");
        return wire;
    }

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = multimask;

    if (out || candidate_out.is_open()) {
        sam3_result first = sam3_segment_pvs(*state, *model, pvs);
        if (out) {
            const auto mask_path = save_detection_mask(output_mask_dir, 0, first);
            write_detection_row(out.get(), 0, 0, first, mask_path);
        }
        write_initial_candidate_rows(candidate_out.is_open() ? &candidate_out : nullptr, first);
    }

    int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    if (inst_id < 0) {
        fail("add_instance failed");
        return wire;
    }

    wire.t_frame0_ms = (ggml_time_us() - t0) / 1000.0;

    // Frames 1..N-1: track / propagate
    double t_track_sum = 0.0;
    std::vector<double> track_times;
    sam3_result last_result;

    for (int f = 1; f < n_frames; f++) {
        t0 = ggml_time_us();

        if (visual_only) {
            last_result = sam3_propagate_frame(*tracker, *state, *model, frames[f]);
        } else {
            last_result = sam3_track_frame(*tracker, *state, *model, frames[f]);
        }
        const auto mask_path = save_detection_mask(output_mask_dir, f, last_result);
        write_detection_row(out.get(), f, f, last_result, mask_path);

        double dt = (ggml_time_us() - t0) / 1000.0;
        t_track_sum += dt;
        track_times.push_back(dt);
        wire.n_track_frames++;

        if (!quiet) {
            fprintf(stderr,
                    "    frame %d/%d  %.0f ms  (%zu det)\n",
                    f,
                    n_frames - 1,
                    dt,
                    last_result.detections.size());
        }
    }

    if (n_frames > 1) {
        wire.t_track_avg_ms = t_track_sum / (n_frames - 1);
    }
    wire.t_track_p50_ms = percentile(track_times, 0.50);
    wire.t_track_p95_ms = percentile(track_times, 0.95);
    wire.t_total_ms = wire.t_frame0_ms + t_track_sum;
    wire.n_detections = (int) last_result.detections.size();
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
                            int n_frames,
                            float px,
                            float py,
                            int n_threads,
                            int encode_img_size,
                            bool bbox_only,
                            bool multimask,
                            int recondition_every,
                            const std::string& output_jsonl,
                            const std::string& output_initial_candidates_jsonl,
                            const std::string& output_mask_dir,
                            bool quiet,
                            int write_fd) {
    BenchWire wire = run_single_benchmark(model_path,
                                          use_gpu,
                                          video_path,
                                          n_frames,
                                          px,
                                          py,
                                          n_threads,
                                          encode_img_size,
                                          bbox_only,
                                          multimask,
                                          recondition_every,
                                          output_jsonl,
                                          output_initial_candidates_jsonl,
                                          output_mask_dir,
                                          quiet);
    const auto bytes = std::as_bytes(std::span{&wire, 1});
    if (!write_full(write_fd, bytes)) {
        fprintf(stderr, "child_benchmark: failed to write result pipe\n");
    }
    close(write_fd);
    _exit(wire.ok ? 0 : 1);
}
#endif

static BenchResult run_benchmark_isolated(const ModelEntry& entry,
                                          bool use_gpu,
                                          const std::string& video_path,
                                          int n_frames,
                                          float px,
                                          float py,
                                          int n_threads,
                                          int encode_img_size = 0,
                                          bool bbox_only = false,
                                          bool multimask = false,
                                          int recondition_every = 16,
                                          const std::string& output_jsonl = "",
                                          const std::string& output_initial_candidates_jsonl = "",
                                          const std::string& output_mask_dir = "",
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
                                          n_frames,
                                          px,
                                          py,
                                          n_threads,
                                          encode_img_size,
                                          bbox_only,
                                          multimask,
                                          recondition_every,
                                          output_jsonl,
                                          output_initial_candidates_jsonl,
                                          output_mask_dir,
                                          quiet);
    if (wire.ok) {
        res.t_load_ms = wire.t_load_ms;
        res.t_frame0_ms = wire.t_frame0_ms;
        res.t_track_avg_ms = wire.t_track_avg_ms;
        res.t_track_p50_ms = wire.t_track_p50_ms;
        res.t_track_p95_ms = wire.t_track_p95_ms;
        res.t_total_ms = wire.t_total_ms;
        res.max_rss_kib = wire.max_rss_kib;
        res.n_detections = wire.n_detections;
        res.backend = display_backend_name(wire.backend.data(), use_gpu);
        res.success = true;
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
                        n_frames,
                        px,
                        py,
                        n_threads,
                        encode_img_size,
                        bbox_only,
                        multimask,
                        recondition_every,
                        output_jsonl,
                        output_initial_candidates_jsonl,
                        output_mask_dir,
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
        res.t_load_ms = wire.t_load_ms;
        res.t_frame0_ms = wire.t_frame0_ms;
        res.t_track_avg_ms = wire.t_track_avg_ms;
        res.t_track_p50_ms = wire.t_track_p50_ms;
        res.t_track_p95_ms = wire.t_track_p95_ms;
        res.t_total_ms = wire.t_total_ms;
        res.max_rss_kib = wire.max_rss_kib;
        res.n_detections = wire.n_detections;
        res.backend = display_backend_name(wire.backend.data(), use_gpu);
        res.success = true;
    } else if (got_wire && !wire.ok) {
        res.error = wire.error.data();
    } else if (WIFSIGNALED(status)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "crashed (signal %d)", WTERMSIG(status));
        res.error = buf;
    } else {
        res.error = "child failed (no result)";
    }
#endif

    return res;
}

static BenchResult run_benchmark_direct(const ModelEntry& entry,
                                        bool use_gpu,
                                        const std::string& video_path,
                                        int n_frames,
                                        float px,
                                        float py,
                                        int n_threads,
                                        int encode_img_size = 0,
                                        bool bbox_only = false,
                                        bool multimask = false,
                                        int recondition_every = 16,
                                        const std::string& output_jsonl = "",
                                        const std::string& output_initial_candidates_jsonl = "",
                                        const std::string& output_mask_dir = "",
                                        bool quiet = false) {
    BenchResult res;
    res.model_name = entry.name;
    res.backend = requested_backend_label(use_gpu);
    res.file_size = entry.file_size;

    BenchWire wire = run_single_benchmark(entry.path,
                                          use_gpu,
                                          video_path,
                                          n_frames,
                                          px,
                                          py,
                                          n_threads,
                                          encode_img_size,
                                          bbox_only,
                                          multimask,
                                          recondition_every,
                                          output_jsonl,
                                          output_initial_candidates_jsonl,
                                          output_mask_dir,
                                          quiet);
    if (wire.ok) {
        res.t_load_ms = wire.t_load_ms;
        res.t_frame0_ms = wire.t_frame0_ms;
        res.t_track_avg_ms = wire.t_track_avg_ms;
        res.t_track_p50_ms = wire.t_track_p50_ms;
        res.t_track_p95_ms = wire.t_track_p95_ms;
        res.t_total_ms = wire.t_total_ms;
        res.max_rss_kib = wire.max_rss_kib;
        res.n_detections = wire.n_detections;
        res.backend = display_backend_name(wire.backend.data(), use_gpu);
        res.success = true;
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
    printf("\n");
    printf(
        "=========================================================================================="
        "===============\n");
    printf("SAM3.CPP BENCHMARK  —  %d frames, point=(%.1f, %.1f), threads=%d\n",
           n_frames,
           px,
           py,
           n_threads);
    printf("video: %s\n", video_path.c_str());
    printf(
        "=========================================================================================="
        "===============\n\n");

    printf("  %3s | %-36s | %7s | %7s | %9s | %9s | %13s | %9s | %9s | %10s | %8s | %3s | %s\n",
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
    printf(
        "------+--------------------------------------+---------+---------+-----------+-----------+"
        "---------------+-----------+-----------+------------+----------+-----+--------\n");

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        if (r.success) {
            printf(
                "  %3d | %-36s | %7s | %7s | %9s | %9s | %13s | %9s | %9s | %10s | %8.1f | %3d | "
                "OK\n",
                (int) (i + 1),
                r.model_name.c_str(),
                format_size(r.file_size).c_str(),
                r.backend.c_str(),
                format_time_short(r.t_load_ms).c_str(),
                format_time_short(r.t_frame0_ms).c_str(),
                format_time_short(r.t_track_avg_ms).c_str(),
                format_time_short(r.t_track_p50_ms).c_str(),
                format_time_short(r.t_track_p95_ms).c_str(),
                format_time_short(r.t_total_ms).c_str(),
                r.max_rss_kib / 1024.0,
                r.n_detections);
        } else {
            printf(
                "  %3d | %-36s | %7s | %7s | %9s | %9s | %13s | %9s | %9s | %10s | %8s | %3s | "
                "FAIL: %s\n",
                (int) (i + 1),
                r.model_name.c_str(),
                format_size(r.file_size).c_str(),
                r.backend.c_str(),
                r.t_load_ms > 0 ? format_time_short(r.t_load_ms).c_str() : "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                r.error.c_str());
        }
    }

    printf(
        "------+--------------------------------------+---------+---------+-----------+-----------+"
        "---------------+-----------+-----------+------------+----------+-----+--------\n");

    int n_ok = 0, n_fail = 0;
    for (const auto& r : results) {
        if (r.success)
            n_ok++;
        else
            n_fail++;
    }
    printf("\nSUMMARY: %d runs, %d OK, %d FAIL\n", (int) results.size(), n_ok, n_fail);
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string models_dir = "models/";
    std::string video_path = "data/test_video.mp4";
    float px = 315.0f;
    float py = 250.0f;
    int n_frames = 10;
    int n_threads = 4;
    int encode_img_size = 0;
    int recondition_every = 16;
    bool cpu_only = false;
    bool gpu_only = false;
    bool bbox_only = false;
    bool multimask = false;
    bool no_isolation = false;
    bool quiet = false;
    std::string filter;
    std::string output_jsonl;
    std::string output_initial_candidates_jsonl;
    std::string output_mask_dir;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--point-x" && i + 1 < argc) {
            px = (float) atof(argv[++i]);
        } else if (arg == "--point-y" && i + 1 < argc) {
            py = (float) atof(argv[++i]);
        } else if (arg == "--n-frames" && i + 1 < argc) {
            n_frames = atoi(argv[++i]);
        } else if (arg == "--n-threads" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (arg == "--encode-img-size" && i + 1 < argc) {
            encode_img_size = atoi(argv[++i]);
        } else if (arg == "--recondition-every" && i + 1 < argc) {
            recondition_every = atoi(argv[++i]);
        } else if (arg == "--cpu-only") {
            cpu_only = true;
        } else if (arg == "--gpu-only") {
            gpu_only = true;
        } else if (arg == "--bbox-only") {
            bbox_only = true;
        } else if (arg == "--multimask") {
            multimask = true;
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
        } else if (arg == "--output-mask-dir" && i + 1 < argc) {
            output_mask_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            fprintf(
                stderr,
                "Usage: %s [options]\n"
                "  --models-dir <path>   Models directory       (default: models/)\n"
                "  --video <path>        Video file             (default: data/test_video.mp4)\n"
                "  --point-x <f>         Click X                (default: 315.0)\n"
                "  --point-y <f>         Click Y                (default: 250.0)\n"
                "  --n-frames <n>        Frames to track        (default: 10)\n"
                "  --n-threads <n>       CPU threads            (default: 4)\n"
                "  --recondition-every <n> Memory refresh interval (default: 16)\n"
                "  --cpu-only            Skip GPU runs\n"
                "  --gpu-only            Skip CPU runs\n"
                "  --bbox-only           Track/output bbox rows without full-res masks\n"
                "  --multimask           Use multimask output for the initial point prompt\n"
                "  --filter <substr>     Filter model filenames\n"
                "  --output-jsonl <path> Write first-run target bbox rows\n"
                "  --output-initial-candidates-jsonl <path> Write initial point-prompt candidates\n"
                "  --output-mask-dir <path> Write first-run target masks as PNG files\n"
                "  --no-isolation        Run in-process for profiler capture\n"
                "  --quiet               Suppress per-frame progress lines\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if ((!output_jsonl.empty() || !output_initial_candidates_jsonl.empty() || !output_mask_dir.empty()) &&
        !(gpu_only || cpu_only)) {
        fprintf(stderr,
                "ERROR: output JSONL/mask options require --gpu-only or --cpu-only to select one "
                "backend\n");
        return 1;
    }

    if (n_frames < 2) {
        fprintf(stderr, "ERROR: --n-frames must be >= 2\n");
        return 1;
    }
    if (recondition_every < 1) {
        fprintf(stderr, "ERROR: --recondition-every must be >= 1\n");
        return 1;
    }

    // Discover models
    auto entries = discover_models(models_dir, filter);
    if (entries.empty()) {
        fprintf(stderr, "ERROR: no .ggml files found in '%s'", models_dir.c_str());
        if (!filter.empty())
            fprintf(stderr, " (filter: '%s')", filter.c_str());
        fprintf(stderr, "\n");
        return 1;
    }

    fprintf(stderr, "Found %zu model(s)\n", entries.size());
    for (const auto& e : entries) {
        fprintf(stderr, "  %s  (%s)\n", e.name.c_str(), format_size(e.file_size).c_str());
    }

    // Validate video
    auto vinfo = sam3_get_video_info(video_path);
    if (vinfo.n_frames <= 0) {
        fprintf(stderr, "ERROR: cannot read video '%s'\n", video_path.c_str());
        return 1;
    }
    if (n_frames > vinfo.n_frames) {
        fprintf(stderr,
                "WARNING: video has %d frames, clamping to %d\n",
                vinfo.n_frames,
                vinfo.n_frames);
        n_frames = vinfo.n_frames;
    }
    fprintf(stderr,
            "Video: %dx%d, %d frames, %.1f fps\n",
            vinfo.width,
            vinfo.height,
            vinfo.n_frames,
            vinfo.fps);

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
        fprintf(stderr, "\nStarting %zu benchmark runs (in-process)...\n\n", runs.size());
    } else {
#ifdef _WIN32
        fprintf(stderr, "\nStarting %zu benchmark runs (in-process)...\n\n", runs.size());
#else
        fprintf(stderr, "\nStarting %zu benchmark runs (each in a subprocess)...\n\n", runs.size());
#endif
    }

    // Run benchmarks
    int64_t t_wall_start = ggml_time_us();
    std::vector<BenchResult> results;
    results.reserve(runs.size());

    for (size_t i = 0; i < runs.size(); i++) {
        const auto& run = runs[i];
        const char* backend_str = requested_backend_label(run.use_gpu);

        fprintf(stderr,
                "[%3zu/%zu] %s (%s) ...\n",
                i + 1,
                runs.size(),
                run.entry->name.c_str(),
                backend_str);

        auto res = no_isolation ? run_benchmark_direct(*run.entry,
                                                       run.use_gpu,
                                                       video_path,
                                                       n_frames,
                                                       px,
                                                       py,
                                                       n_threads,
                                                       encode_img_size,
                                                       bbox_only,
                                                       multimask,
                                                       recondition_every,
                                                       (i == 0) ? output_jsonl : "",
                                                       (i == 0) ? output_initial_candidates_jsonl : "",
                                                       (i == 0) ? output_mask_dir : "",
                                                       quiet)
                                : run_benchmark_isolated(*run.entry,
                                                         run.use_gpu,
                                                         video_path,
                                                         n_frames,
                                                         px,
                                                         py,
                                                         n_threads,
                                                         encode_img_size,
                                                         bbox_only,
                                                         multimask,
                                                         recondition_every,
                                                         (i == 0) ? output_jsonl : "",
                                                         (i == 0) ? output_initial_candidates_jsonl : "",
                                                         (i == 0) ? output_mask_dir : "",
                                                         quiet);
        results.push_back(res);

        if (res.success) {
            fprintf(stderr,
                    "  -> OK  backend=%s  load=%.0fms  init=%.0fms  track/fr=%.0fms  p50=%.0fms  "
                    "p95=%.0fms  total=%.0fms  rss=%.1fMiB  det=%d\n\n",
                    res.backend.c_str(),
                    res.t_load_ms,
                    res.t_frame0_ms,
                    res.t_track_avg_ms,
                    res.t_track_p50_ms,
                    res.t_track_p95_ms,
                    res.t_total_ms,
                    res.max_rss_kib / 1024.0,
                    res.n_detections);
        } else {
            fprintf(stderr, "  -> FAIL: %s\n\n", res.error.c_str());
        }
    }

    double t_wall_ms = (ggml_time_us() - t_wall_start) / 1000.0;

    // Print table
    print_table(results, video_path, px, py, n_frames, n_threads);

    double t_wall_s = t_wall_ms / 1000.0;
    int mins = (int) (t_wall_s / 60.0);
    int secs = (int) (t_wall_s) % 60;
    printf("Total wall time: %dm %ds\n\n", mins, secs);

    return 0;
}
