#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

inline constexpr int SAM3_VERSION_MAJOR = 1;
inline constexpr int SAM3_VERSION_MINOR = 0;
inline constexpr int SAM3_VERSION_PATCH = 0;
#define SAM3_VERSION "1.0.0"

/*
** ── Forward Declarations ─────────────────────────────────────────────────
*/

struct sam3_model;
struct sam3_state;
struct sam3_tracker;

/* Custom deleters so unique_ptr works with forward-declared opaque types. */
struct sam3_state_deleter {
    void operator()(sam3_state* p) const;
};
struct sam3_tracker_deleter {
    void operator()(sam3_tracker* p) const;
};

using sam3_state_ptr = std::unique_ptr<sam3_state, sam3_state_deleter>;
using sam3_tracker_ptr = std::unique_ptr<sam3_tracker, sam3_tracker_deleter>;

/*
** ── Model Type ──────────────────────────────────────────────────────────
*/

enum class sam3_model_type {
    sam3 = 0,         // Full SAM3 (ViT + detector + tracker)
    sam3_visual = 1,  // SAM3 visual-only (ViT + tracker, no text)
    sam2 = 2,         // SAM2 (Hiera + tracker, no text/detector)
    edgetam = 3,      // EdgeTAM (RepViT + Perceiver, no text/detector)
};

inline constexpr sam3_model_type SAM3_MODEL_SAM3 = sam3_model_type::sam3;
inline constexpr sam3_model_type SAM3_MODEL_SAM3_VISUAL = sam3_model_type::sam3_visual;
inline constexpr sam3_model_type SAM3_MODEL_SAM2 = sam3_model_type::sam2;
inline constexpr sam3_model_type SAM3_MODEL_EDGETAM = sam3_model_type::edgetam;

/*****************************************************************************
** Public Data Types
**
** Geometry primitives, images, masks, and detection results.
*****************************************************************************/

struct sam3_point {
    float x;
    float y;
};

struct sam3_box {
    float x0;  // top-left x
    float y0;  // top-left y
    float x1;  // bottom-right x
    float y1;  // bottom-right y
};

struct sam3_image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

struct sam3_mask {
    int width = 0;
    int height = 0;
    float iou_score = 0.0f;
    float obj_score = 0.0f;
    int instance_id = -1;
    std::vector<uint8_t> data;  // binary mask (0 or 255)
};

struct sam3_detection {
    sam3_box box;
    float score = 0.0f;
    float iou_score = 0.0f;
    int instance_id = -1;
    sam3_mask mask;
    std::vector<float> sam_token;  // raw SAM decoder output token (for obj_ptr)
};

struct sam3_result {
    std::vector<sam3_detection> detections;
};

/*****************************************************************************
** Parameters
**
** Configuration for model loading, segmentation, and video tracking.
*****************************************************************************/

struct sam3_params {
    std::string model_path;
    int n_threads = 4;
    bool use_gpu = true;
    int seed = 42;
    int encode_img_size = 0;  // 0 = model default; override input resolution
};

struct sam3_tensor_info {
    int64_t ne[4] = {0, 0, 0, 0};
    uint64_t nb[4] = {0, 0, 0, 0};
    int type = 0;
    int op = 0;
    bool is_contiguous = false;
};

enum class sam3_vit_block_stage {
    norm1 = 0,
    window_part,
    qkv_proj,
    attn_core,
    attn_proj,
    window_unpart,
    norm2,
    mlp_fc1,
    mlp_gelu,
    mlp_fc2,
    mlp,
};

inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_NORM1 = sam3_vit_block_stage::norm1;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_WINDOW_PART =
    sam3_vit_block_stage::window_part;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_QKV_PROJ =
    sam3_vit_block_stage::qkv_proj;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_ATTN_CORE =
    sam3_vit_block_stage::attn_core;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_ATTN_PROJ =
    sam3_vit_block_stage::attn_proj;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART =
    sam3_vit_block_stage::window_unpart;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_NORM2 = sam3_vit_block_stage::norm2;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_MLP_FC1 = sam3_vit_block_stage::mlp_fc1;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_MLP_GELU =
    sam3_vit_block_stage::mlp_gelu;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_MLP_FC2 = sam3_vit_block_stage::mlp_fc2;
inline constexpr sam3_vit_block_stage SAM3_VIT_BLOCK_STAGE_MLP = sam3_vit_block_stage::mlp;

enum class sam3_vit_prefix_stage {
    patch_embed = 0,
    patch_im2col,
    patch_mulmat_raw,
    patch_mulmat,
    pos_add,
    ln_pre_norm,
    ln_pre,
};

inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_PATCH_EMBED =
    sam3_vit_prefix_stage::patch_embed;
inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL =
    sam3_vit_prefix_stage::patch_im2col;
inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW =
    sam3_vit_prefix_stage::patch_mulmat_raw;
inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT =
    sam3_vit_prefix_stage::patch_mulmat;
inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_POS_ADD =
    sam3_vit_prefix_stage::pos_add;
inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM =
    sam3_vit_prefix_stage::ln_pre_norm;
inline constexpr sam3_vit_prefix_stage SAM3_VIT_PREFIX_STAGE_LN_PRE = sam3_vit_prefix_stage::ln_pre;

struct sam3_pcs_params {
    std::string text_prompt;
    std::vector<sam3_box> pos_exemplars;
    std::vector<sam3_box> neg_exemplars;
    float score_threshold = 0.5f;
    float nms_threshold = 0.1f;
};

struct sam3_pvs_params {
    std::vector<sam3_point> pos_points;
    std::vector<sam3_point> neg_points;
    sam3_box box = {
        .x0 = 0.0f,
        .y0 = 0.0f,
        .x1 = 0.0f,
        .y1 = 0.0f,
    };
    bool use_box = false;
    bool multimask = false;
};

struct sam3_video_params {
    std::string text_prompt;
    float score_threshold = 0.5f;
    float nms_threshold = 0.1f;
    float assoc_iou_threshold = 0.1f;
    int hotstart_delay = 15;
    int max_keep_alive = 30;
    int recondition_every = 16;
    int fill_hole_area = 16;
    bool bbox_only = false;
};

struct sam3_video_info {
    int width = 0;
    int height = 0;
    int n_frames = 0;
    float fps = 0.0f;
};

/*****************************************************************************
** Public API
**
** Model lifecycle, image encoding, segmentation, and video tracking.
*****************************************************************************/

/*
** ── Model Lifecycle ──────────────────────────────────────────────────────
*/

/*
** Load a SAM3 model from the file specified in params.model_path.
** Returns nullptr on failure.
*/
std::shared_ptr<sam3_model> sam3_load_model(const sam3_params& params);

/* Free all resources held by a loaded model. */
void sam3_free_model(sam3_model& model);

/* Returns true if the model was loaded as visual-only (no text/detector path).
** SAM2 models are always considered visual-only. */
bool sam3_is_visual_only(const sam3_model& model);

/* Returns the model type (SAM2 or SAM3). */
sam3_model_type sam3_get_model_type(const sam3_model& model);

/* Returns the active ggml backend name, e.g. "CUDA0", "Metal", or "CPU". */
const char* sam3_backend_name(const sam3_model& model);

/*
** ── Inference State ──────────────────────────────────────────────────────
*/

/* Allocate inference state (backbone caches, PE buffers). */
sam3_state_ptr sam3_create_state(const sam3_model& model, const sam3_params& params);

/* Free inference state and its GPU buffers. */
void sam3_free_state(sam3_state& state);

/*
** ── Image Backbone ───────────────────────────────────────────────────────
*/

/*
** Encode an image through the ViT backbone and FPN neck.
** Call once per image before segmentation or tracking.
** Returns true on success, false on failure.
*/
bool sam3_encode_image(sam3_state& state, const sam3_model& model, const sam3_image& image);

/*
** ── Image Segmentation ──────────────────────────────────────────────────
*/

/* Segment using text prompt + exemplar boxes (PCS path). */
sam3_result sam3_segment_pcs(sam3_state& state,
                             const sam3_model& model,
                             const sam3_pcs_params& params);

/* Segment using point/box prompts (PVS path). */
sam3_result sam3_segment_pvs(sam3_state& state,
                             const sam3_model& model,
                             const sam3_pvs_params& params);

/*
** ── Video Tracking ──────────────────────────────────────────────────────
*/

/* Create a tracker for text-prompted video segmentation. */
sam3_tracker_ptr sam3_create_tracker(const sam3_model& model, const sam3_video_params& params);

/* Encode a frame, detect objects, and update tracked instances. */
sam3_result sam3_track_frame(sam3_tracker& tracker,
                             sam3_state& state,
                             const sam3_model& model,
                             const sam3_image& frame);

/* Refine a tracked instance with interactive point prompts. */
bool sam3_refine_instance(sam3_tracker& tracker,
                          sam3_state& state,
                          const sam3_model& model,
                          int instance_id,
                          const std::vector<sam3_point>& pos_points,
                          const std::vector<sam3_point>& neg_points);

/*
** Add a new instance to the tracker from PVS prompts (points/box) on the
** current frame.  The image must already be encoded (via sam3_track_frame
** or sam3_encode_image).  Returns assigned instance_id, or -1 on failure.
*/
int sam3_tracker_add_instance(sam3_tracker& tracker,
                              sam3_state& state,
                              const sam3_model& model,
                              const sam3_pvs_params& pvs_params);

/* Return the current frame index of the tracker. */
int sam3_tracker_frame_index(const sam3_tracker& tracker);

/* Reset the tracker, clearing all instances and memory. */
void sam3_tracker_reset(sam3_tracker& tracker);

/*
** ── Visual-Only Video Tracking ──────────────────────────────────────────
*/

struct sam3_visual_track_params {
    float assoc_iou_threshold = 0.1f;
    int max_keep_alive = 30;
    int recondition_every = 16;
    int fill_hole_area = 16;
    bool bbox_only = false;
};

/*
** Create a tracker for visual-only models.  Instances are added manually
** via sam3_tracker_add_instance().
*/
sam3_tracker_ptr sam3_create_visual_tracker(const sam3_model& model,
                                            const sam3_visual_track_params& params);

/*
** Propagate all tracked instances to the next frame (no detection step).
** The image is encoded, then each tracked instance is propagated via
** memory attention + SAM mask decode, and the memory bank is updated.
*/
sam3_result sam3_propagate_frame(sam3_tracker& tracker,
                                 sam3_state& state,
                                 const sam3_model& model,
                                 const sam3_image& frame);

/*
** ── Utility ─────────────────────────────────────────────────────────────
*/

sam3_image sam3_load_image(std::string_view path);
bool sam3_save_mask(const sam3_mask& mask, std::string_view path);
sam3_image sam3_decode_video_frame(std::string_view video_path, int frame_index);
sam3_video_info sam3_get_video_info(std::string_view video_path);

/*
** ── C ABI ───────────────────────────────────────────────────────────────
**
** Thin opaque-handle API for language bindings. All functions return null,
** false, or -1 on failure and never throw C++ exceptions across the ABI.
*/

extern "C" {

struct sam3c_model;
struct sam3c_state;
struct sam3c_tracker;
struct sam3c_result;

struct sam3c_box {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct sam3c_detection {
    sam3c_box box;
    float score;
    float iou_score;
    int instance_id;
    int mask_width;
    int mask_height;
    const uint8_t* mask_data;
    size_t mask_len;
};

sam3c_model* sam3c_load_model(const char* model_path,
                              int n_threads,
                              int use_gpu,
                              int encode_img_size);
void sam3c_free_model(sam3c_model* model);
const char* sam3c_backend_name(const sam3c_model* model);

sam3c_state* sam3c_create_state(const sam3c_model* model);
void sam3c_free_state(sam3c_state* state);

sam3c_tracker* sam3c_create_visual_tracker(const sam3c_model* model,
                                           int recondition_every,
                                           int bbox_only);
void sam3c_free_tracker(sam3c_tracker* tracker);

bool sam3c_encode_image_rgb(sam3c_state* state,
                            const sam3c_model* model,
                            int width,
                            int height,
                            const uint8_t* data,
                            size_t len);

sam3c_result* sam3c_segment_point(sam3c_state* state, const sam3c_model* model, float x, float y);
sam3c_result* sam3c_segment_box(
    sam3c_state* state, const sam3c_model* model, float x0, float y0, float x1, float y1);

int sam3c_add_point_instance(
    sam3c_tracker* tracker, sam3c_state* state, const sam3c_model* model, float x, float y);
int sam3c_add_box_instance(sam3c_tracker* tracker,
                           sam3c_state* state,
                           const sam3c_model* model,
                           float x0,
                           float y0,
                           float x1,
                           float y1);

sam3c_result* sam3c_propagate_frame_rgb(sam3c_tracker* tracker,
                                        sam3c_state* state,
                                        const sam3c_model* model,
                                        int width,
                                        int height,
                                        const uint8_t* data,
                                        size_t len);

size_t sam3c_result_count(const sam3c_result* result);
bool sam3c_result_detection(const sam3c_result* result,
                            size_t index,
                            sam3c_detection* out_detection);
void sam3c_free_result(sam3c_result* result);
}

/*****************************************************************************
** Test and Debug API
**
** Standalone tokenizer, intermediate tensor dumps, and debug utilities.
** These functions are intended for testing and development only.
*****************************************************************************/

bool sam3_test_load_tokenizer(const std::string& model_path);
std::vector<int32_t> sam3_test_tokenize(const std::string& text);

/*
** Run the text encoder on fixed token IDs and dump standard intermediate
** tensors to <output_dir>/<tensor_name>.{bin,shape}.
*/
bool sam3_test_dump_text_encoder(const sam3_model& model,
                                 const std::vector<int32_t>& token_ids,
                                 const std::string& output_dir,
                                 int n_threads = 4);

/*
** Run the full phase 5 detector path (fusion encoder + DETR decoder +
** dot-product scoring + segmentation head) on an already-encoded image
** and dump intermediate tensors.
*/
bool sam3_test_dump_phase5(const sam3_model& model,
                           const sam3_state& state,
                           const std::vector<int32_t>& token_ids,
                           const std::string& output_dir,
                           int n_threads = 4);

/*
** Run the phase 5 detector from pre-dumped inputs instead of re-running
** the image/text encoders.  Isolates detector numerics from earlier phases.
*/
bool sam3_test_dump_phase5_from_ref_inputs(const sam3_model& model,
                                           const std::vector<int32_t>& token_ids,
                                           const std::string& prephase_ref_dir,
                                           const std::string& phase5_ref_dir,
                                           const std::string& output_dir,
                                           int n_threads = 4);

/*
** Run the phase 6 prompt encoder + SAM decoder on an already-encoded
** tracker image state and dump intermediate tensors.
*/
bool sam3_test_dump_phase6(const sam3_model& model,
                           const sam3_state& state,
                           const sam3_pvs_params& params,
                           const std::string& output_dir,
                           int n_threads = 4);

/*
** Run the phase 6 prompt encoder + SAM decoder from pre-dumped phase 3
** tracker features.  Isolates phase 6 numerics from earlier phases.
*/
bool sam3_test_dump_phase6_from_ref_inputs(const sam3_model& model,
                                           const std::string& prephase_ref_dir,
                                           const sam3_pvs_params& params,
                                           const std::string& output_dir,
                                           int n_threads = 4);

/*
** Run the phase 7 tracker subgraph from pre-dumped case inputs and dump
** intermediate tensors.  Case directory produced by dump_phase7_reference.py.
*/
bool sam3_test_dump_phase7_from_ref_inputs(const sam3_model& model,
                                           const std::string& case_ref_dir,
                                           const std::string& output_dir,
                                           int n_threads = 4);

/*
** Run the geometry encoder from pre-computed backbone features and dump
** intermediate tensors.  Tests exemplar box coordinate encoding against
** Python reference.
*/
bool sam3_test_dump_geom_enc(const sam3_model& model,
                             const std::string& prephase_ref_dir,
                             const sam3_pcs_params& params,
                             const std::string& output_dir,
                             int n_threads = 4);

/*
** Run ONLY the fusion encoder (6 layers) from pre-dumped inputs (image
** features, pos encoding, prompt tokens, attn bias).  Dumps per-layer
** outputs for isolated fenc debugging.
*/
bool sam3_test_fenc_only(const sam3_model& model,
                         const std::string& ref_dir,
                         const std::string& output_dir,
                         int n_threads = 4);

/*
** ── Debug ────────────────────────────────────────────────────────────────
*/

/* Dump a named state tensor to a binary file for verification. */
bool sam3_dump_state_tensor(const sam3_state& state,
                            const std::string& tensor_name,
                            const std::string& output_path);

/* Query metadata for a named state tensor without dumping its payload. */
bool sam3_get_state_tensor_info(const sam3_state& state,
                                const std::string& tensor_name,
                                sam3_tensor_info& info);

/* Dump a named model tensor (weights/constants) to a binary file. */
bool sam3_dump_model_tensor(const sam3_model& model,
                            const std::string& tensor_name,
                            const std::string& output_path);

/* Query metadata for a named model tensor. */
bool sam3_get_model_tensor_info(const sam3_model& model,
                                const std::string& tensor_name,
                                sam3_tensor_info& info);

/*
** Encode an image from pre-preprocessed float data (CHW layout, already
** resized and normalized).  Bypasses C++ preprocessing so that numerical
** comparisons against the Python reference are not polluted by resize
** implementation differences.
*/
bool sam3_encode_image_from_preprocessed(sam3_state& state,
                                         const sam3_model& model,
                                         const float* chw_data,
                                         int img_size);

/*
** Test-only: run ONLY the ViT encoder from preprocessed float data and keep
** only the requested intermediate tensors alive for dumping/comparison.
*/
bool sam3_encode_vit_from_preprocessed_selective(sam3_state& state,
                                                 const sam3_model& model,
                                                 const float* chw_data,
                                                 int img_size,
                                                 const std::vector<std::string>& output_tensors);

/*
** Test-only: run the exact ViT prefix up to the tensor entering block 0
** (patch embed + pos embed + ln_pre).
*/
bool sam3_test_run_vit_block0_input(const sam3_model& model,
                                    const float* chw_data,
                                    int img_size,
                                    std::vector<float>& output_data,
                                    int64_t output_ne[4],
                                    int n_threads = 4);

/*
** Test-only: run an exact ViT prefix sub-stage on the real model tensors using
** the model's backend. PATCH_EMBED expects image input [W,H,3,1]. Later stages
** expect feature input [E,W,H,1] in ggml 4D layout.
*/
bool sam3_test_run_vit_prefix_stage(const sam3_model& model,
                                    sam3_vit_prefix_stage stage,
                                    std::span<const float> input_data,
                                    std::array<int64_t, 4> input_ne,
                                    std::vector<float>& output_data,
                                    int64_t output_ne[4],
                                    int n_threads = 4);
bool sam3_test_run_patch_mulmat_host_ref(const sam3_model& model,
                                         std::span<const float> input_data,
                                         std::array<int64_t, 4> input_ne,
                                         bool use_double_accum,
                                         std::vector<float>& output_data,
                                         int64_t output_ne[4]);
bool sam3_test_run_vit_block_linear_host_ref(const sam3_model& model,
                                             int block_idx,
                                             sam3_vit_block_stage stage,
                                             std::span<const float> input_data,
                                             std::array<int64_t, 4> input_ne,
                                             bool use_double_accum,
                                             std::vector<float>& output_data,
                                             int64_t output_ne[4]);

/*
** Test-only: run an exact ViT block sub-stage on the real model tensors using
** the model's backend. The input is always provided as F32 in ggml 4D layout.
*/
bool sam3_test_run_vit_block_stage(const sam3_model& model,
                                   int block_idx,
                                   sam3_vit_block_stage stage,
                                   std::span<const float> input_data,
                                   std::array<int64_t, 4> input_ne,
                                   std::vector<float>& output_data,
                                   int64_t output_ne[4],
                                   int n_threads = 4);

/*
** ── Profiling ───────────────────────────────────────────────────────────
*/

/*
** Profile the EdgeTAM image encoder (RepViT backbone + FPN neck).
**
** Runs the full graph once for a total timing and op summary, then builds
** and times each stage as a separate sub-graph to produce a per-stage
** latency breakdown:
**   - Stem (2 convolutions)
**   - Stage 0..3 (downsample + RepViT blocks)
**   - FPN neck (lateral convolutions + top-down fusion)
**
** n_warmup iterations are run before n_iter timed iterations.
** Results are printed to stderr.
*/
bool sam3_profile_edgetam_encode(const sam3_model& model,
                                 const sam3_image& image,
                                 int n_threads = 4,
                                 int n_warmup = 2,
                                 int n_iter = 5);
