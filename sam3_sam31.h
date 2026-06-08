#pragma once

namespace sam3::sam31 {

inline constexpr int image_size = 1008;
inline constexpr int backbone_stride = 14;
inline constexpr int multiplex_count = 16;
inline constexpr int eval_multiplex_count = 16;
inline constexpr int max_num_objects = 16;
inline constexpr int num_maskmem = 7;
inline constexpr int max_obj_ptrs_in_encoder = 16;
inline constexpr int tracker_mem_dim = 256;

struct MaskDecoderContract {
    int embedding_dim;
    int multiplex_count;
    int iou_token_count;
    int obj_score_token_count;
    int mask_token_count;
    int masks_per_object;
    int output_token_count_before_sparse_prompts;
    int hyper_mlp_count;
    int twoway_depth;
    bool has_obj_score_token;
    bool uses_high_res_features;
};

inline constexpr MaskDecoderContract propagation_mask_decoder = {
    .embedding_dim = 256,
    .multiplex_count = multiplex_count,
    .iou_token_count = 16,
    .obj_score_token_count = 16,
    .mask_token_count = 48,
    .masks_per_object = 3,
    .output_token_count_before_sparse_prompts = 80,
    .hyper_mlp_count = 3,
    .twoway_depth = 2,
    .has_obj_score_token = true,
    .uses_high_res_features = true,
};

inline constexpr MaskDecoderContract interactive_mask_decoder = {
    .embedding_dim = 256,
    .multiplex_count = 1,
    .iou_token_count = 1,
    .obj_score_token_count = 1,
    .mask_token_count = 4,
    .masks_per_object = 4,
    .output_token_count_before_sparse_prompts = 6,
    .hyper_mlp_count = 4,
    .twoway_depth = 2,
    .has_obj_score_token = true,
    .uses_high_res_features = true,
};

static_assert(propagation_mask_decoder.multiplex_count == multiplex_count);
static_assert(propagation_mask_decoder.mask_token_count == 48);
static_assert(propagation_mask_decoder.masks_per_object == 3);
static_assert(propagation_mask_decoder.output_token_count_before_sparse_prompts == 80);

}  // namespace sam3::sam31
