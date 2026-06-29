set shell := ["bash", "-cu"]
set dotenv-load

BUILD_MODE := env_var_or_default("BUILD_MODE", "release")
CUDA_ARCH := env_var_or_default("CUDA_ARCH", "120")
CXX_COMPILER := env_var_or_default("CXX_COMPILER", "/usr/bin/g++-14")
CUDA_HOST_COMPILER := env_var_or_default("CUDA_HOST_COMPILER", "/usr/bin/g++-14")
CUDA_FORCE_CUBLAS := env_var_or_default("CUDA_FORCE_CUBLAS", "n")
CUDA_FORCE_MMQ := env_var_or_default("CUDA_FORCE_MMQ", "n")
MODELS_DIR := env_var_or_default("SAM3_MODELS_DIR", "models")
VIDEO := env_var_or_default("SAM3_VIDEO", "data/test_video.mp4")
PARITY_RUNS := env_var_or_default("PARITY_RUNS", "15")
PARITY_OUT := env_var_or_default("PARITY_OUT", "outputs/parity-stats")
TRACK_JSONL_LHS := env_var_or_default("TRACK_JSONL_LHS", "outputs/base-plus-parity/cpu.jsonl")
TRACK_JSONL_RHS := env_var_or_default("TRACK_JSONL_RHS", "outputs/base-plus-parity/cuda.jsonl")
TRACK_JSONL_OUT := env_var_or_default("TRACK_JSONL_OUT", "outputs/base-plus-parity/cpu-vs-cuda-summary.json")
PROFILE_LOGS := env_var_or_default("PROFILE_LOGS", "outputs/hiera-cuda-profile/*.log")
PROFILE_SUMMARY_OUT := env_var_or_default("PROFILE_SUMMARY_OUT", "outputs/hiera-cuda-profile/summary.json")
SAM2_REPO := env_var_or_default("SAM2_REPO", "../sam2")
SAM2_BASE_PLUS_CHECKPOINT := env_var_or_default("SAM2_BASE_PLUS_CHECKPOINT", "../sam2/checkpoints/sam2.1_hiera_base_plus.pt")
SAM2_QUALITY_CPP_JSONL := env_var_or_default("SAM2_QUALITY_CPP_JSONL", "outputs/sam2-official-quality-10/cpp.jsonl")
SAM2_QUALITY_OUT := env_var_or_default("SAM2_QUALITY_OUT", "outputs/sam2-official-quality-10")
SAM2_QUALITY_IMAGE_SIZE := env_var_or_default("SAM2_QUALITY_IMAGE_SIZE", "0")
FUSION_OUT := env_var_or_default("FUSION_OUT", "outputs/fusion-kernel-acceptance")
FUSION_TYPE := env_var_or_default("FUSION_TYPE", "q4_1")
FUSION_TOLERANCE := env_var_or_default("FUSION_TOLERANCE", "10")
FUSION_REPEATS := env_var_or_default("FUSION_REPEATS", "5")
MODEL_MATRIX_OUT := env_var_or_default("MODEL_MATRIX_OUT", "outputs/model-matrix-compare")
MODEL_MATRIX_FRAMES := env_var_or_default("MODEL_MATRIX_FRAMES", "10")
MODEL_MATRIX_REPEATS := env_var_or_default("MODEL_MATRIX_REPEATS", "3")
MODEL_MATRIX_FILTER := env_var_or_default("MODEL_MATRIX_FILTER", "")
MODEL_MATRIX_TIMED_START_FRAME := env_var_or_default("MODEL_MATRIX_TIMED_START_FRAME", "1")
MODEL_MATRIX_ENCODE_IMG_SIZE := env_var_or_default("MODEL_MATRIX_ENCODE_IMG_SIZE", "")
MODEL_MATRIX_PYTHON_FAMILY := env_var_or_default("MODEL_MATRIX_PYTHON_FAMILY", "")
MODEL_MATRIX_PYTHON_SAM3_VERSION := env_var_or_default("MODEL_MATRIX_PYTHON_SAM3_VERSION", "none")
MODEL_MATRIX_PYTHON_DTYPE := env_var_or_default("MODEL_MATRIX_PYTHON_DTYPE", "bf16")
MODEL_MATRIX_TF32_POLICY := env_var_or_default("MODEL_MATRIX_TF32_POLICY", "on")
MODEL_MATRIX_SKIP_PYTHON_SAM2 := env_var_or_default("MODEL_MATRIX_SKIP_PYTHON_SAM2", "")
MODEL_MATRIX_CPP_PREENCODE_TRACK_FRAMES := env_var_or_default("MODEL_MATRIX_CPP_PREENCODE_TRACK_FRAMES", "")
MODEL_MATRIX_CPP_PREENCODE_CACHED_TAIL_FRAMES := env_var_or_default("MODEL_MATRIX_CPP_PREENCODE_CACHED_TAIL_FRAMES", "")
MODEL_MATRIX_CPP_TEXT_INIT_SELECTED_ONLY := env_var_or_default("MODEL_MATRIX_CPP_TEXT_INIT_SELECTED_ONLY", "")
MODEL_MATRIX_CPP_NO_OUTPUT_ARTIFACTS := env_var_or_default("MODEL_MATRIX_CPP_NO_OUTPUT_ARTIFACTS", "")
MODEL_MATRIX_CPP_MASK_OUTPUT := env_var_or_default("MODEL_MATRIX_CPP_MASK_OUTPUT", "")
MODEL_MATRIX_CPP_FRAME_TIMING_DIR := env_var_or_default("MODEL_MATRIX_CPP_FRAME_TIMING_DIR", "")
MODEL_MATRIX_CPP_VARIANT := env_var_or_default("MODEL_MATRIX_CPP_VARIANT", "default")
MODEL_MATRIX_CPP_ENV := env_var_or_default("MODEL_MATRIX_CPP_ENV", "")
MODEL_MATRIX_CPP_WARMUP_RUNS := env_var_or_default("MODEL_MATRIX_CPP_WARMUP_RUNS", "2")
MODEL_MATRIX_PYTHON_RESULTS := env_var_or_default("MODEL_MATRIX_PYTHON_RESULTS", "")
E2E_REQUIRED_SPLIT_OUT := env_var_or_default("E2E_REQUIRED_SPLIT_OUT", "outputs/e2e-required-split")
E2E_REQUIRED_SPLIT_FRAMES := env_var_or_default("E2E_REQUIRED_SPLIT_FRAMES", "5")
E2E_REQUIRED_SPLIT_REPEATS := env_var_or_default("E2E_REQUIRED_SPLIT_REPEATS", "3")
E2E_REQUIRED_SPLIT_MODELS_DIR := env_var_or_default("E2E_REQUIRED_SPLIT_MODELS_DIR", "models/sam3")
E2E_REQUIRED_SPLIT_FILTER := env_var_or_default("E2E_REQUIRED_SPLIT_FILTER", "sam3-bf16")
E2E_REQUIRED_SPLIT_PYTHON_SAM3_VERSION := env_var_or_default("E2E_REQUIRED_SPLIT_PYTHON_SAM3_VERSION", "sam3")
E2E_REQUIRED_SPLIT_PYTHON_DTYPE := env_var_or_default("E2E_REQUIRED_SPLIT_PYTHON_DTYPE", "bf16")
E2E_REQUIRED_SPLIT_TF32_POLICY := env_var_or_default("E2E_REQUIRED_SPLIT_TF32_POLICY", "on")
E2E_REQUIRED_SPLIT_CPP_VARIANT := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_VARIANT", "default")
E2E_REQUIRED_SPLIT_CPP_ENV := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_ENV", "")
E2E_REQUIRED_SPLIT_CPP_WARMUP_RUNS := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_WARMUP_RUNS", "2")
E2E_REQUIRED_SPLIT_PYTHON_RESULTS := env_var_or_default("E2E_REQUIRED_SPLIT_PYTHON_RESULTS", "")
E2E_REQUIRED_SPLIT_CPP_PREENCODE_CACHED_TAIL_FRAMES := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_PREENCODE_CACHED_TAIL_FRAMES", "1")
E2E_REQUIRED_SPLIT_CPP_TEXT_INIT_SELECTED_ONLY := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_TEXT_INIT_SELECTED_ONLY", "1")
E2E_REQUIRED_SPLIT_CPP_NO_OUTPUT_ARTIFACTS := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_NO_OUTPUT_ARTIFACTS", "1")
E2E_REQUIRED_SPLIT_CPP_MASK_OUTPUT := env_var_or_default("E2E_REQUIRED_SPLIT_CPP_MASK_OUTPUT", "")
E2E_REQUIRED_PROFILE_OUT := env_var_or_default("E2E_REQUIRED_PROFILE_OUT", "outputs/e2e-required-profile")
E2E_REQUIRED_PROFILE_FRAMES_DIR := env_var_or_default("E2E_REQUIRED_PROFILE_FRAMES_DIR", "outputs/e2e-required-split/frames")
E2E_REQUIRED_PROFILE_FRAMES := env_var_or_default("E2E_REQUIRED_PROFILE_FRAMES", "3")
E2E_REQUIRED_PROFILE_MODELS_DIR := env_var_or_default("E2E_REQUIRED_PROFILE_MODELS_DIR", "models/sam3")
E2E_REQUIRED_PROFILE_FILTER := env_var_or_default("E2E_REQUIRED_PROFILE_FILTER", "sam3-bf16")
E2E_REQUIRED_PROFILE_CPP_ENV := env_var_or_default("E2E_REQUIRED_PROFILE_CPP_ENV", "")
E2E_REQUIRED_PROFILE_TIMED_START_FRAME := env_var_or_default("E2E_REQUIRED_PROFILE_TIMED_START_FRAME", "1")
E2E_REQUIRED_PROFILE_CPP_WARMUP_RUNS := env_var_or_default("E2E_REQUIRED_PROFILE_CPP_WARMUP_RUNS", "0")
E2E_REQUIRED_PROFILE_CPP_PREENCODE_CACHED_TAIL_FRAMES := env_var_or_default("E2E_REQUIRED_PROFILE_CPP_PREENCODE_CACHED_TAIL_FRAMES", "1")
E2E_REQUIRED_PROFILE_CPP_TEXT_INIT_SELECTED_ONLY := env_var_or_default("E2E_REQUIRED_PROFILE_CPP_TEXT_INIT_SELECTED_ONLY", "1")
E2E_REQUIRED_PROFILE_CPP_NO_OUTPUT_ARTIFACTS := env_var_or_default("E2E_REQUIRED_PROFILE_CPP_NO_OUTPUT_ARTIFACTS", "1")
E2E_REQUIRED_MULMAT_PROFILE_OUT := env_var_or_default("E2E_REQUIRED_MULMAT_PROFILE_OUT", "outputs/e2e-required-mulmat-profile")
E2E_REQUIRED_VIT_BENCH_OUT := env_var_or_default("E2E_REQUIRED_VIT_BENCH_OUT", "outputs/e2e-required-vit-bench")
E2E_REQUIRED_VIT_BENCH_MODEL := env_var_or_default("E2E_REQUIRED_VIT_BENCH_MODEL", "models/sam3/sam3-bf16.ggml")
E2E_REQUIRED_VIT_BENCH_IMAGE := env_var_or_default("E2E_REQUIRED_VIT_BENCH_IMAGE", "outputs/e2e-required-split/frames/00001.jpg")
E2E_REQUIRED_VIT_BENCH_BATCH_SIZE := env_var_or_default("E2E_REQUIRED_VIT_BENCH_BATCH_SIZE", "1")
E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS := env_var_or_default("E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS", "")
E2E_REQUIRED_VIT_BENCH_BLOCK0_STAGE := env_var_or_default("E2E_REQUIRED_VIT_BENCH_BLOCK0_STAGE", "")
E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE := env_var_or_default("E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE", E2E_REQUIRED_VIT_BENCH_BLOCK0_STAGE)
E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE_INDEX := env_var_or_default("E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE_INDEX", "0")
E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS := env_var_or_default("E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS", "3")
E2E_REQUIRED_VIT_BENCH_REPEATS := env_var_or_default("E2E_REQUIRED_VIT_BENCH_REPEATS", "10")
E2E_REQUIRED_VIT_BENCH_CPP_ENV := env_var_or_default("E2E_REQUIRED_VIT_BENCH_CPP_ENV", "")
E2E_REQUIRED_VIT_BENCH_WITH_TRACKER_NECK := env_var_or_default("E2E_REQUIRED_VIT_BENCH_WITH_TRACKER_NECK", "1")
E2E_REQUIRED_VIT_STAGE_SWEEP_OUT := env_var_or_default("E2E_REQUIRED_VIT_STAGE_SWEEP_OUT", "outputs/e2e-required-vit-stage-sweep")
E2E_REQUIRED_VIT_STAGE_SWEEP_STAGES := env_var_or_default("E2E_REQUIRED_VIT_STAGE_SWEEP_STAGES", "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14")
E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_OUT := env_var_or_default("E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_OUT", "outputs/e2e-required-vit-isolated-stage-sweep")
E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCK := env_var_or_default("E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCK", E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE_INDEX)
E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCKS := env_var_or_default("E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCKS", "0 7")
E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_STAGES := env_var_or_default("E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_STAGES", E2E_REQUIRED_VIT_STAGE_SWEEP_STAGES)
E2E_REQUIRED_VIT_STAGE_AB_OUT := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_OUT", "outputs/e2e-required-vit-stage-ab")
E2E_REQUIRED_VIT_STAGE_AB_BASELINE := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_BASELINE", "outputs/e2e-required-vit-isolated-stage-sweep/default/summary.json")
E2E_REQUIRED_VIT_STAGE_AB_CANDIDATE := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_CANDIDATE", "outputs/e2e-required-vit-isolated-stage-sweep/candidate/summary.json")
E2E_REQUIRED_VIT_STAGE_AB_STAGE := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_STAGE", "")
E2E_REQUIRED_VIT_STAGE_AB_STAGE_NAME := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_STAGE_NAME", "")
E2E_REQUIRED_VIT_STAGE_AB_AUDIT := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_AUDIT", "outputs/e2e-required-audit/optimization_targets.json")
E2E_REQUIRED_VIT_STAGE_AB_BUDGET_SCOPE := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_BUDGET_SCOPE", "tail")
E2E_REQUIRED_VIT_STAGE_AB_DROP_INITIAL_SAMPLES := env_var_or_default("E2E_REQUIRED_VIT_STAGE_AB_DROP_INITIAL_SAMPLES", "1")
E2E_REQUIRED_VIT_BLOCK_SWEEP_OUT := env_var_or_default("E2E_REQUIRED_VIT_BLOCK_SWEEP_OUT", "outputs/e2e-required-vit-block-sweep")
E2E_REQUIRED_VIT_BLOCK_SWEEP_BLOCKS := env_var_or_default("E2E_REQUIRED_VIT_BLOCK_SWEEP_BLOCKS", "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32")
E2E_REQUIRED_VIT_BLOCK_SWEEP_WITH_NECK := env_var_or_default("E2E_REQUIRED_VIT_BLOCK_SWEEP_WITH_NECK", "1")
E2E_REQUIRED_MEASURE_OUT := env_var_or_default("E2E_REQUIRED_MEASURE_OUT", "outputs/e2e-required-measure")
E2E_REQUIRED_MEASURE_ISOLATED_STAGE_SWEEP := env_var_or_default("E2E_REQUIRED_MEASURE_ISOLATED_STAGE_SWEEP", "1")
E2E_REQUIRED_AUDIT_OUT := env_var_or_default("E2E_REQUIRED_AUDIT_OUT", "outputs/e2e-required-audit")
E2E_REQUIRED_AUDIT_COMPONENT_SPLIT := env_var_or_default("E2E_REQUIRED_AUDIT_COMPONENT_SPLIT", "outputs/e2e-required-split-sam3-bf16-neck-names-r3-20260628a/component_split.json")
E2E_REQUIRED_AUDIT_CANDIDATE_COMPONENT_SPLITS := env_var_or_default("E2E_REQUIRED_AUDIT_CANDIDATE_COMPONENT_SPLITS", "")
E2E_REQUIRED_AUDIT_CANDIDATE_PARITY := env_var_or_default("E2E_REQUIRED_AUDIT_CANDIDATE_PARITY", "")
E2E_REQUIRED_AUDIT_PYTHON_PARITY := env_var_or_default("E2E_REQUIRED_AUDIT_PYTHON_PARITY", "")
E2E_REQUIRED_AUDIT_ISOLATED_BENCHES := env_var_or_default("E2E_REQUIRED_AUDIT_ISOLATED_BENCHES", "")
E2E_REQUIRED_AUDIT_FATTN_PROFILES := env_var_or_default("E2E_REQUIRED_AUDIT_FATTN_PROFILES", "")
E2E_REQUIRED_AUDIT_NODE_NAMES := env_var_or_default("E2E_REQUIRED_AUDIT_NODE_NAMES", "outputs/e2e-required-profile-neck-names-20260628a/node_names_split.json")
E2E_REQUIRED_AUDIT_NODE_LABEL := env_var_or_default("E2E_REQUIRED_AUDIT_NODE_LABEL", "sam3_encode#02")
E2E_REQUIRED_AUDIT_STAGE_PROFILE := env_var_or_default("E2E_REQUIRED_AUDIT_STAGE_PROFILE", "")
E2E_REQUIRED_AUDIT_CUBLASLT := env_var_or_default("E2E_REQUIRED_AUDIT_CUBLASLT", "outputs/e2e-required-mulmat-profile-sam3-bf16-baseline-20260628b/cublaslt_bias_by_group.json")
E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_OUT := env_var_or_default("E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_OUT", "outputs/e2e-required-cublaslt-algo-sweep")
E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_COMPONENT_SPLIT := env_var_or_default("E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_COMPONENT_SPLIT", E2E_REQUIRED_AUDIT_COMPONENT_SPLIT)
E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_NODE_NAMES := env_var_or_default("E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_NODE_NAMES", E2E_REQUIRED_AUDIT_NODE_NAMES)
E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_CUBLASLT := env_var_or_default("E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_CUBLASLT", E2E_REQUIRED_AUDIT_CUBLASLT)
E2E_REQUIRED_AB_OUT := env_var_or_default("E2E_REQUIRED_AB_OUT", "outputs/e2e-required-ab")
E2E_REQUIRED_AB_BASELINE_VARIANT := env_var_or_default("E2E_REQUIRED_AB_BASELINE_VARIANT", "baseline")
E2E_REQUIRED_AB_BASELINE_CPP_ENV := env_var_or_default("E2E_REQUIRED_AB_BASELINE_CPP_ENV", "")
E2E_REQUIRED_AB_CANDIDATE_VARIANT := env_var_or_default("E2E_REQUIRED_AB_CANDIDATE_VARIANT", "candidate")
E2E_REQUIRED_AB_CANDIDATE_CPP_ENV := env_var_or_default("E2E_REQUIRED_AB_CANDIDATE_CPP_ENV", "")
E2E_REQUIRED_PARITY_OUT := env_var_or_default("E2E_REQUIRED_PARITY_OUT", "outputs/e2e-required-parity")
E2E_REQUIRED_PARITY_FRAMES_DIR := env_var_or_default("E2E_REQUIRED_PARITY_FRAMES_DIR", "outputs/e2e-required-split/frames")
E2E_REQUIRED_PARITY_FRAMES := env_var_or_default("E2E_REQUIRED_PARITY_FRAMES", "5")
E2E_REQUIRED_PARITY_MODELS_DIR := env_var_or_default("E2E_REQUIRED_PARITY_MODELS_DIR", "models/sam3")
E2E_REQUIRED_PARITY_FILTER := env_var_or_default("E2E_REQUIRED_PARITY_FILTER", "sam3-bf16")
E2E_REQUIRED_PARITY_BASELINE_VARIANT := env_var_or_default("E2E_REQUIRED_PARITY_BASELINE_VARIANT", "baseline")
E2E_REQUIRED_PARITY_BASELINE_CPP_ENV := env_var_or_default("E2E_REQUIRED_PARITY_BASELINE_CPP_ENV", "")
E2E_REQUIRED_PARITY_CANDIDATE_VARIANT := env_var_or_default("E2E_REQUIRED_PARITY_CANDIDATE_VARIANT", "candidate")
E2E_REQUIRED_PARITY_CANDIDATE_CPP_ENV := env_var_or_default("E2E_REQUIRED_PARITY_CANDIDATE_CPP_ENV", "")
E2E_REQUIRED_PARITY_CPP_WARMUP_RUNS := env_var_or_default("E2E_REQUIRED_PARITY_CPP_WARMUP_RUNS", "2")
E2E_REQUIRED_PARITY_TIMED_START_FRAME := env_var_or_default("E2E_REQUIRED_PARITY_TIMED_START_FRAME", "1")
E2E_REQUIRED_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES := env_var_or_default("E2E_REQUIRED_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES", "1")
E2E_REQUIRED_PARITY_CPP_TEXT_INIT_SELECTED_ONLY := env_var_or_default("E2E_REQUIRED_PARITY_CPP_TEXT_INIT_SELECTED_ONLY", "1")
E2E_REQUIRED_PARITY_CPP_MASK_OUTPUT := env_var_or_default("E2E_REQUIRED_PARITY_CPP_MASK_OUTPUT", "")
E2E_REQUIRED_PYTHON_PARITY_OUT := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_OUT", "outputs/e2e-required-python-parity")
E2E_REQUIRED_PYTHON_PARITY_FRAMES_DIR := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_FRAMES_DIR", E2E_REQUIRED_PARITY_FRAMES_DIR)
E2E_REQUIRED_PYTHON_PARITY_FRAMES := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_FRAMES", E2E_REQUIRED_PARITY_FRAMES)
E2E_REQUIRED_PYTHON_PARITY_MODELS_DIR := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_MODELS_DIR", E2E_REQUIRED_PARITY_MODELS_DIR)
E2E_REQUIRED_PYTHON_PARITY_FILTER := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_FILTER", E2E_REQUIRED_PARITY_FILTER)
E2E_REQUIRED_PYTHON_PARITY_CPP_ENV := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_CPP_ENV", "")
E2E_REQUIRED_PYTHON_PARITY_CPP_WARMUP_RUNS := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_CPP_WARMUP_RUNS", E2E_REQUIRED_PARITY_CPP_WARMUP_RUNS)
E2E_REQUIRED_PYTHON_PARITY_TIMED_START_FRAME := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_TIMED_START_FRAME", E2E_REQUIRED_PARITY_TIMED_START_FRAME)
E2E_REQUIRED_PYTHON_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES", E2E_REQUIRED_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES)
E2E_REQUIRED_PYTHON_PARITY_CPP_TEXT_INIT_SELECTED_ONLY := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_CPP_TEXT_INIT_SELECTED_ONLY", E2E_REQUIRED_PARITY_CPP_TEXT_INIT_SELECTED_ONLY)
E2E_REQUIRED_PYTHON_PARITY_DTYPE := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_DTYPE", E2E_REQUIRED_SPLIT_PYTHON_DTYPE)
E2E_REQUIRED_PYTHON_PARITY_TF32_POLICY := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_TF32_POLICY", E2E_REQUIRED_SPLIT_TF32_POLICY)
E2E_REQUIRED_PYTHON_PARITY_VERSION := env_var_or_default("E2E_REQUIRED_PYTHON_PARITY_VERSION", "sam3")
SAM3_REPO := env_var_or_default("SAM3_REPO", "external/sam3")
CUDNN_SDPA_OUT := env_var_or_default("CUDNN_SDPA_OUT", "outputs/cudnn-sdpa-head64")
CUDNN_SDPA_MODEL_FILTER := env_var_or_default("CUDNN_SDPA_MODEL_FILTER", "sam3-f16")
CUDA_HEALTH_OUT := env_var_or_default("CUDA_HEALTH_OUT", "outputs/cuda-health")
SAM31_CONTRACT_OUT := env_var_or_default("SAM31_CONTRACT_OUT", "outputs/sam31-checkpoint-contract")
SAM31_CONTRACT_SAM3 := env_var_or_default("SAM31_CONTRACT_SAM3", "models/sam3/sam3.pt")
SAM31_CONTRACT_SAM31 := env_var_or_default("SAM31_CONTRACT_SAM31", "models/sam3.1/sam3.1_multiplex.pt")
SAM31_SLICE := env_var_or_default("SAM31_SLICE", "multiplex_mask_decoder")
SAM31_MULTIPLEX_STATE_OUT := env_var_or_default("SAM31_MULTIPLEX_STATE_OUT", "outputs/sam31-multiplex-state")
SAM31_MUX_MASK_DECODER_OUT := env_var_or_default("SAM31_MUX_MASK_DECODER_OUT", "outputs/sam31-mux-mask-decoder-case")
SAM31_MUX_MASK_DECODER_CPP_OUT := env_var_or_default("SAM31_MUX_MASK_DECODER_CPP_OUT", "outputs/sam31-mux-mask-decoder-cpp")
SAM31_MUX_MASK_DECODER_COMPARE_OUT := env_var_or_default("SAM31_MUX_MASK_DECODER_COMPARE_OUT", "outputs/sam31-mux-mask-decoder-compare")
SAM31_MUX_MASK_DECODER_MODEL := env_var_or_default("SAM31_MUX_MASK_DECODER_MODEL", "models/sam3.1/sam3.1_mux_mask_decoder-f32.ggml")
SAM31_MUX_MASK_DECODER_FEAT_SIZE := env_var_or_default("SAM31_MUX_MASK_DECODER_FEAT_SIZE", "4")
SAM31_MUX_MASK_DECODER_BUCKETS := env_var_or_default("SAM31_MUX_MASK_DECODER_BUCKETS", "1")
SAM31_MEMORY_BACKBONE_OUT := env_var_or_default("SAM31_MEMORY_BACKBONE_OUT", "outputs/sam31-memory-backbone-case")
SAM31_MEMORY_BACKBONE_CPP_OUT := env_var_or_default("SAM31_MEMORY_BACKBONE_CPP_OUT", "outputs/sam31-memory-backbone-cpp")
SAM31_MEMORY_BACKBONE_COMPARE_OUT := env_var_or_default("SAM31_MEMORY_BACKBONE_COMPARE_OUT", "outputs/sam31-memory-backbone-compare")
SAM31_MEMORY_BACKBONE_MODEL := env_var_or_default("SAM31_MEMORY_BACKBONE_MODEL", "models/sam3.1/sam3.1_memory_backbone-f32.ggml")
SAM31_MEMORY_BACKBONE_FEAT_SIZE := env_var_or_default("SAM31_MEMORY_BACKBONE_FEAT_SIZE", "72")
SAM31_MEMORY_BACKBONE_MASK_SIZE := env_var_or_default("SAM31_MEMORY_BACKBONE_MASK_SIZE", "1152")
SAM31_MEMORY_BACKBONE_BUCKETS := env_var_or_default("SAM31_MEMORY_BACKBONE_BUCKETS", "1")
SAM31_MEMORY_BACKBONE_DEVICE := env_var_or_default("SAM31_MEMORY_BACKBONE_DEVICE", "gpu")
SAM31_MEMORY_BACKBONE_MAX_ABS_TOLERANCE := env_var_or_default("SAM31_MEMORY_BACKBONE_MAX_ABS_TOLERANCE", "1e-2")
SAM31_MEMORY_BACKBONE_MEAN_ABS_TOLERANCE := env_var_or_default("SAM31_MEMORY_BACKBONE_MEAN_ABS_TOLERANCE", "1e-3")
SAM31_PROPAGATION_FEATURES_OUT := env_var_or_default("SAM31_PROPAGATION_FEATURES_OUT", "outputs/sam31-propagation-features-case")
SAM31_PROPAGATION_FEATURES_CPP_OUT := env_var_or_default("SAM31_PROPAGATION_FEATURES_CPP_OUT", "outputs/sam31-propagation-features-cpp")
SAM31_PROPAGATION_FEATURES_COMPARE_OUT := env_var_or_default("SAM31_PROPAGATION_FEATURES_COMPARE_OUT", "outputs/sam31-propagation-features-compare")
SAM31_PROPAGATION_FEATURES_MODEL := env_var_or_default("SAM31_PROPAGATION_FEATURES_MODEL", "models/sam3.1/sam3.1_propagation_features-f32.ggml")
SAM31_PROPAGATION_FEATURES_FEAT_SIZE := env_var_or_default("SAM31_PROPAGATION_FEATURES_FEAT_SIZE", "8")
SAM31_PROPAGATION_FEATURES_DEVICE := env_var_or_default("SAM31_PROPAGATION_FEATURES_DEVICE", "cpu")
SAM31_PROPAGATION_FEATURES_MAX_ABS_TOLERANCE := env_var_or_default("SAM31_PROPAGATION_FEATURES_MAX_ABS_TOLERANCE", "1e-2")
SAM31_PROPAGATION_FEATURES_MEAN_ABS_TOLERANCE := env_var_or_default("SAM31_PROPAGATION_FEATURES_MEAN_ABS_TOLERANCE", "1e-3")
SAM31_MEMORY_ATTENTION_OUT := env_var_or_default("SAM31_MEMORY_ATTENTION_OUT", "outputs/sam31-memory-attention-case")
SAM31_MEMORY_ATTENTION_CPP_OUT := env_var_or_default("SAM31_MEMORY_ATTENTION_CPP_OUT", "outputs/sam31-memory-attention-cpp")
SAM31_MEMORY_ATTENTION_COMPARE_OUT := env_var_or_default("SAM31_MEMORY_ATTENTION_COMPARE_OUT", "outputs/sam31-memory-attention-compare")
SAM31_MEMORY_ATTENTION_MODEL := env_var_or_default("SAM31_MEMORY_ATTENTION_MODEL", "models/sam3.1/sam3.1_memory_attention-f16.ggml")
SAM31_MEMORY_ATTENTION_FTYPE := env_var_or_default("SAM31_MEMORY_ATTENTION_FTYPE", "1")
SAM31_MEMORY_ATTENTION_DTYPE := env_var_or_default("SAM31_MEMORY_ATTENTION_DTYPE", "auto")
SAM31_MEMORY_ATTENTION_PY_DEVICE := env_var_or_default("SAM31_MEMORY_ATTENTION_PY_DEVICE", "auto")
SAM31_MEMORY_ATTENTION_FEAT_SIZE := env_var_or_default("SAM31_MEMORY_ATTENTION_FEAT_SIZE", "72")
SAM31_MEMORY_ATTENTION_BUCKETS := env_var_or_default("SAM31_MEMORY_ATTENTION_BUCKETS", "1")
SAM31_MEMORY_ATTENTION_DEVICE := env_var_or_default("SAM31_MEMORY_ATTENTION_DEVICE", "gpu")
SAM31_MEMORY_ATTENTION_MAX_ABS_TOLERANCE := env_var_or_default("SAM31_MEMORY_ATTENTION_MAX_ABS_TOLERANCE", "4.5e-1")
SAM31_MEMORY_ATTENTION_MEAN_ABS_TOLERANCE := env_var_or_default("SAM31_MEMORY_ATTENTION_MEAN_ABS_TOLERANCE", "2e-2")
SAM31_MEMORY_ATTENTION_DECODER_OUT := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_OUT", "outputs/sam31-memory-attention-decoder-case")
SAM31_MEMORY_ATTENTION_DECODER_CPP_OUT := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_CPP_OUT", "outputs/sam31-memory-attention-decoder-cpp")
SAM31_MEMORY_ATTENTION_DECODER_COMPARE_OUT := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_COMPARE_OUT", "outputs/sam31-memory-attention-decoder-compare")
SAM31_MEMORY_ATTENTION_DECODER_MODEL := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_MODEL", "models/sam3.1/sam3.1_memory_attention_decoder-f32.ggml")
SAM31_MEMORY_ATTENTION_DECODER_FTYPE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_FTYPE", "0")
SAM31_MEMORY_ATTENTION_DECODER_DTYPE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_DTYPE", "f32")
SAM31_MEMORY_ATTENTION_DECODER_PY_DEVICE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_PY_DEVICE", "cpu")
SAM31_MEMORY_ATTENTION_DECODER_FEAT_SIZE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_FEAT_SIZE", "4")
SAM31_MEMORY_ATTENTION_DECODER_BUCKETS := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_BUCKETS", "1")
SAM31_MEMORY_ATTENTION_DECODER_DEVICE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_DEVICE", "cpu")
SAM31_MEMORY_ATTENTION_DECODER_MAX_ABS_TOLERANCE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_MAX_ABS_TOLERANCE", "1e-2")
SAM31_MEMORY_ATTENTION_DECODER_MEAN_ABS_TOLERANCE := env_var_or_default("SAM31_MEMORY_ATTENTION_DECODER_MEAN_ABS_TOLERANCE", "1e-3")
SAM31_TRACKING_STATE_OUT := env_var_or_default("SAM31_TRACKING_STATE_OUT", "outputs/sam31-tracking-state-inspect")
SAM31_TRACKING_STATE_FRAMES := env_var_or_default("SAM31_TRACKING_STATE_FRAMES", "outputs/model-matrix-sam3-bf16-cuda-preprocess-pe-cache/frames")
SAM31_TRACKING_STATE_PROMPT_MODE := env_var_or_default("SAM31_TRACKING_STATE_PROMPT_MODE", "box")
SAM31_TRACKING_STATE_PROMPT := env_var_or_default("SAM31_TRACKING_STATE_PROMPT", "person")
SAM31_TRACKING_STATE_BOX_XYWH := env_var_or_default("SAM31_TRACKING_STATE_BOX_XYWH", "0.25 0.25 0.5 0.5")
SAM31_TRACKING_STATE_BOX_LABEL := env_var_or_default("SAM31_TRACKING_STATE_BOX_LABEL", "1")
SAM31_TRACKING_STATE_STREAM_FRAMES := env_var_or_default("SAM31_TRACKING_STATE_STREAM_FRAMES", "2")
SAM31_TRACKING_MASK_INIT_MODEL := env_var_or_default("SAM31_TRACKING_MASK_INIT_MODEL", "models/sam3.1/sam3.1_multiplex-bf16.ggml")
SAM31_TRACKING_MASK_INIT_DEVICE := env_var_or_default("SAM31_TRACKING_MASK_INIT_DEVICE", "gpu")
SAM31_TRACKING_MASK_INIT_THREADS := env_var_or_default("SAM31_TRACKING_MASK_INIT_THREADS", "4")
SAM31_TRACKING_MASK_INIT_WIDTH := env_var_or_default("SAM31_TRACKING_MASK_INIT_WIDTH", "320")
SAM31_TRACKING_MASK_INIT_HEIGHT := env_var_or_default("SAM31_TRACKING_MASK_INIT_HEIGHT", "240")
SAM31_TRACKING_MASK_INIT_CASE := env_var_or_default("SAM31_TRACKING_MASK_INIT_CASE", "center")
SAM31_TRACKING_MASK_INIT_FRAME1_OFFSET := env_var_or_default("SAM31_TRACKING_MASK_INIT_FRAME1_OFFSET", "3")
SAM31_TRACKING_MASK_INIT_WARMUP_RUNS := env_var_or_default("SAM31_TRACKING_MASK_INIT_WARMUP_RUNS", "0")
SAM31_TRACKING_MASK_INIT_OUT := env_var_or_default("SAM31_TRACKING_MASK_INIT_OUT", "outputs/sam31-tracking-mask-init-smoke")
SAM31_MASK_SEQUENCE_PY_OUT := env_var_or_default("SAM31_MASK_SEQUENCE_PY_OUT", "outputs/sam31-mask-sequence-python")
SAM31_MASK_SEQUENCE_PY_DTYPE := env_var_or_default("SAM31_MASK_SEQUENCE_PY_DTYPE", "bf16")
SAM31_MASK_SEQUENCE_PY_TF32 := env_var_or_default("SAM31_MASK_SEQUENCE_PY_TF32", "on")
SAM31_MASK_SEQUENCE_PY_WARMUP_RUNS := env_var_or_default("SAM31_MASK_SEQUENCE_PY_WARMUP_RUNS", "0")
SAM31_MASK_INIT_MATRIX_OUT := env_var_or_default("SAM31_MASK_INIT_MATRIX_OUT", "outputs/sam31-mask-init-matrix")
SAM31_MASK_INIT_MATRIX_SIZES := env_var_or_default("SAM31_MASK_INIT_MATRIX_SIZES", "320x240")
SAM31_MASK_INIT_MATRIX_CASES := env_var_or_default("SAM31_MASK_INIT_MATRIX_CASES", "center@3")
SAM31_MASK_INIT_MATRIX_VARIANTS := env_var_or_default("SAM31_MASK_INIT_MATRIX_VARIANTS", "default,all,sa123")
SAM31_MASK_INIT_MATRIX_REPEATS := env_var_or_default("SAM31_MASK_INIT_MATRIX_REPEATS", "3")
SAM31_MASK_INIT_MATRIX_PY_WARMUP_RUNS := env_var_or_default("SAM31_MASK_INIT_MATRIX_PY_WARMUP_RUNS", "1")
SAM31_MASK_INIT_MATRIX_CPP_WARMUP_RUNS := env_var_or_default("SAM31_MASK_INIT_MATRIX_CPP_WARMUP_RUNS", "1")
SAM31_MASK_INIT_MATRIX_INTERLEAVE := env_var_or_default("SAM31_MASK_INIT_MATRIX_INTERLEAVE", "1")
SAM31_MASK_INIT_MATRIX_PROFILE_DEFAULT := env_var_or_default("SAM31_MASK_INIT_MATRIX_PROFILE_DEFAULT", "0")
SAM31_MASK_INIT_MATRIX_PROFILE_WARMUP_RUNS := env_var_or_default("SAM31_MASK_INIT_MATRIX_PROFILE_WARMUP_RUNS", "")
SAM31_MASK_INIT_MATRIX_PROFILE_STAGE_TOP := env_var_or_default("SAM31_MASK_INIT_MATRIX_PROFILE_STAGE_TOP", "16")
SAM31_MASK_INIT_MATRIX_PROFILE_NODE_TOP := env_var_or_default("SAM31_MASK_INIT_MATRIX_PROFILE_NODE_TOP", "80")
SAM31_MASK_INIT_AUDIT_OUT := env_var_or_default("SAM31_MASK_INIT_AUDIT_OUT", "outputs/sam31-mask-init-audit")
SAM31_MASK_INIT_STATE_PY_OUT := env_var_or_default("SAM31_MASK_INIT_STATE_PY_OUT", "outputs/sam31-mask-init-state-python")
SAM31_MASK_INIT_STATE_CPP_OUT := env_var_or_default("SAM31_MASK_INIT_STATE_CPP_OUT", "outputs/sam31-mask-init-state-cpp")
SAM31_MASK_INIT_STATE_COMPARE_OUT := env_var_or_default("SAM31_MASK_INIT_STATE_COMPARE_OUT", "outputs/sam31-mask-init-state-compare")
SAM3_SAM31_GOAL_AUDIT_OUT := env_var_or_default("SAM3_SAM31_GOAL_AUDIT_OUT", "outputs/sam3-sam31-goal-audit")
SAM3_PERF_SUMMARY := env_var_or_default("SAM3_PERF_SUMMARY", "outputs/model-matrix-sam3-f16-winpart-vec4-r3/summary.json")
SAM3_PERF_SUMMARY_EXTRA_ARGS := env_var_or_default("SAM3_PERF_SUMMARY_EXTRA_ARGS", "")
SAM3_E2E_AUDIT := env_var_or_default("SAM3_E2E_AUDIT", "outputs/e2e-required-audit/optimization_targets.json")

CLANG_TIDY := env_var_or_default("CLANG_TIDY", "uvx --from clang-tidy==22.1.0 clang-tidy")
CLANG_FORMAT := env_var_or_default("CLANG_FORMAT", "uvx --from clang-format==22.1.5 clang-format")
CPPCHECK := env_var_or_default("CPPCHECK", "uvx --from cppcheck==1.5.1 cppcheck")

# Default: list all recipes.
default:
    @just --list

# Configure xmake for the CUDA benchmark build.
config mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} --cuda_force_cublas={{ CUDA_FORCE_CUBLAS }} --cuda_force_mmq={{ CUDA_FORCE_MMQ }} -y

# Build sam3_benchmark via xmake.
build mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} --cuda_force_cublas={{ CUDA_FORCE_CUBLAS }} --cuda_force_mmq={{ CUDA_FORCE_MMQ }} -y
    xmake build sam3_benchmark

# Build a specific xmake target.
build-target target mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} --cuda_force_cublas={{ CUDA_FORCE_CUBLAS }} --cuda_force_mmq={{ CUDA_FORCE_MMQ }} -y
    xmake build {{ target }}

# Remove xmake/CMake build artifacts.
clean:
    xmake clean

# Full distclean.
distclean:
    xmake f -c
    rm -rf build .xmake

# Generate compile_commands.json for clang-tidy/cppcheck.
compile-db mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} --cuda_force_cublas={{ CUDA_FORCE_CUBLAS }} --cuda_force_mmq={{ CUDA_FORCE_MMQ }} -y
    xmake build sam3_benchmark
    suffix=cuda; \
    if [ "{{ CUDA_FORCE_CUBLAS }}" = "y" ]; then suffix=cuda-force-cublas; fi; \
    if [ "{{ CUDA_FORCE_MMQ }}" = "y" ]; then suffix=cuda-force-mmq; fi; \
    ln -sf build/xmake-{{ mode }}-$suffix/compile_commands.json compile_commands.json

# Format files changed in the current worktree/index.
fmt:
    files="$({ git diff --name-only --diff-filter=ACMR; git diff --cached --name-only --diff-filter=ACMR; } | sort -u | grep -E '^(sam3(_multiplex|_sam31|_sam31_tensor_manifest)?\.(cpp|h)|examples/.*\.(cpp|h))$' || true)"; \
    if [ -n "$files" ]; then printf '%s\n' "$files" | xargs -r {{ CLANG_FORMAT }} -i; fi

# Format all project-owned C++ sources.
fmt-all:
    find sam3.cpp sam3.h sam3_multiplex.h sam3_sam31.h sam3_sam31_tensor_manifest.h examples -type f \( -name '*.cpp' -o -name '*.h' \) | xargs -r {{ CLANG_FORMAT }} -i

# Check formatting without modifying files.
fmt-check:
    find sam3.cpp sam3.h sam3_multiplex.h sam3_sam31.h sam3_sam31_tensor_manifest.h examples -type f \( -name '*.cpp' -o -name '*.h' \) | xargs -r {{ CLANG_FORMAT }} --dry-run -Werror

# Clang-Tidy on one file.
tidy file: compile-db
    {{ CLANG_TIDY }} -p . {{ file }}

# Clang-Tidy on project-owned .cpp files that are buildable without optional GUI deps.
tidy-all: compile-db
    printf '%s\0' sam3.cpp examples/benchmark.cpp examples/profile_edgetam.cpp examples/quantize.cpp examples/smoke_sam3.cpp | xargs -0 -n1 -P$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu) {{ CLANG_TIDY }} -p . --quiet

# Cppcheck on project-owned code.
cppcheck:
    {{ CPPCHECK }} \
        --enable=warning,performance,portability \
        --std=c++23 --language=c++ \
        --suppressions-list=.cppcheck-suppressions --inline-suppr \
        -I . -I ggml/include -I stb \
        --quiet --error-exitcode=1 \
        -j$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu) \
        sam3.cpp sam3_multiplex.h sam3_sam31.h sam3_sam31_tensor_manifest.h examples/benchmark.cpp examples/profile_edgetam.cpp examples/quantize.cpp examples/smoke_sam3.cpp examples/sam31_multiplex_state.cpp examples/sam31_mux_mask_decoder_case.cpp examples/sam31_memory_attention_case.cpp examples/sam31_memory_attention_decoder_case.cpp

# Full local static-analysis pipeline.
lint: fmt-check tidy-all cppcheck

# Run the EdgeTAM parity and paired performance statistics.
parity-stats runs=PARITY_RUNS out=PARITY_OUT: build
    uv run --no-project scripts/sam3_parity_stats.py \
        --benchmark build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir {{ MODELS_DIR }} \
        --video {{ VIDEO }} \
        --runs {{ runs }} \
        --out-dir {{ out }}

# Compare two benchmark JSONL tracking outputs.
compare-tracking-jsonl lhs=TRACK_JSONL_LHS rhs=TRACK_JSONL_RHS out=TRACK_JSONL_OUT:
    uv run scripts/compare_tracking_jsonl.py {{ lhs }} {{ rhs }} --out {{ out }}

# Summarize SAM3_PROFILE benchmark logs.
summarize-profile logs=PROFILE_LOGS out=PROFILE_SUMMARY_OUT:
    uv run scripts/summarize_benchmark_profile.py {{ logs }} --out {{ out }}

# Compare C++ SAM2 tracking masks with official PyTorch SAM2 masks.
compare-sam2-official-quality cpp_jsonl=SAM2_QUALITY_CPP_JSONL out=SAM2_QUALITY_OUT:
    uv run scripts/compare_sam2_official_quality.py \
        --repo . \
        --video {{ VIDEO }} \
        --cpp-jsonl {{ cpp_jsonl }} \
        --sam2-repo {{ SAM2_REPO }} \
        --checkpoint {{ SAM2_BASE_PLUS_CHECKPOINT }} \
        --image-size {{ SAM2_QUALITY_IMAGE_SIZE }} \
        --out-dir {{ out }}

# Compare C++ models with official Python SAM2/SAM3 baselines under one input contract.
model-matrix out=MODEL_MATRIX_OUT:
    uv run --no-project --with huggingface_hub --with pillow scripts/model_matrix_compare.py \
        --repo . \
        --models-dir {{ MODELS_DIR }} \
        --video {{ VIDEO }} \
        --out-dir {{ out }} \
        --frames {{ MODEL_MATRIX_FRAMES }} \
        --repeats {{ MODEL_MATRIX_REPEATS }} \
        --timed-start-frame {{ MODEL_MATRIX_TIMED_START_FRAME }} \
        --python-dtype {{ MODEL_MATRIX_PYTHON_DTYPE }} \
        --tf32-policy {{ MODEL_MATRIX_TF32_POLICY }} \
        --python-sam3-version {{ MODEL_MATRIX_PYTHON_SAM3_VERSION }} \
        --cpp-warmup-runs {{ MODEL_MATRIX_CPP_WARMUP_RUNS }} \
        --cpp-variant {{ MODEL_MATRIX_CPP_VARIANT }} \
        ${MODEL_MATRIX_CPP_ENV:+--cpp-env "$MODEL_MATRIX_CPP_ENV"} \
        ${MODEL_MATRIX_ENCODE_IMG_SIZE:+--encode-img-size "$MODEL_MATRIX_ENCODE_IMG_SIZE"} \
        ${MODEL_MATRIX_CPP_PREENCODE_TRACK_FRAMES:+--cpp-preencode-track-frames} \
        ${MODEL_MATRIX_CPP_PREENCODE_CACHED_TAIL_FRAMES:+--cpp-preencode-cached-tail-frames} \
        ${MODEL_MATRIX_CPP_TEXT_INIT_SELECTED_ONLY:+--cpp-text-init-selected-only} \
        ${MODEL_MATRIX_CPP_NO_OUTPUT_ARTIFACTS:+--cpp-no-output-artifacts} \
        ${MODEL_MATRIX_CPP_MASK_OUTPUT:+--cpp-mask-output} \
        ${MODEL_MATRIX_CPP_FRAME_TIMING_DIR:+--cpp-frame-timing-dir "$MODEL_MATRIX_CPP_FRAME_TIMING_DIR"} \
        ${MODEL_MATRIX_PYTHON_RESULTS:+--python-results "$MODEL_MATRIX_PYTHON_RESULTS"} \
        ${MODEL_MATRIX_SKIP_PYTHON_SAM2:+--skip-python-sam2} \
        ${MODEL_MATRIX_PYTHON_FAMILY:+--python-family "$MODEL_MATRIX_PYTHON_FAMILY"} \
        ${MODEL_MATRIX_FILTER:+--filter "$MODEL_MATRIX_FILTER"}

# Run the same-contract SAM3 required-E2E split used for optimization A/B decisions.
e2e-required-split out=E2E_REQUIRED_SPLIT_OUT: build
    mkdir -p "{{ out }}/cpp-frame-timing"
    MODEL_MATRIX_OUT="{{ out }}" \
    SAM3_MODELS_DIR="{{ E2E_REQUIRED_SPLIT_MODELS_DIR }}" \
    MODEL_MATRIX_FRAMES="{{ E2E_REQUIRED_SPLIT_FRAMES }}" \
    MODEL_MATRIX_REPEATS="{{ E2E_REQUIRED_SPLIT_REPEATS }}" \
    MODEL_MATRIX_FILTER="{{ E2E_REQUIRED_SPLIT_FILTER }}" \
    MODEL_MATRIX_PYTHON_SAM3_VERSION="{{ E2E_REQUIRED_SPLIT_PYTHON_SAM3_VERSION }}" \
    MODEL_MATRIX_PYTHON_DTYPE="{{ E2E_REQUIRED_SPLIT_PYTHON_DTYPE }}" \
    MODEL_MATRIX_TF32_POLICY="{{ E2E_REQUIRED_SPLIT_TF32_POLICY }}" \
    MODEL_MATRIX_SKIP_PYTHON_SAM2=1 \
    MODEL_MATRIX_CPP_PREENCODE_CACHED_TAIL_FRAMES="{{ E2E_REQUIRED_SPLIT_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" \
    MODEL_MATRIX_CPP_TEXT_INIT_SELECTED_ONLY="{{ E2E_REQUIRED_SPLIT_CPP_TEXT_INIT_SELECTED_ONLY }}" \
    MODEL_MATRIX_CPP_NO_OUTPUT_ARTIFACTS="{{ E2E_REQUIRED_SPLIT_CPP_NO_OUTPUT_ARTIFACTS }}" \
    MODEL_MATRIX_CPP_MASK_OUTPUT="{{ E2E_REQUIRED_SPLIT_CPP_MASK_OUTPUT }}" \
    MODEL_MATRIX_CPP_FRAME_TIMING_DIR="cpp-frame-timing" \
    MODEL_MATRIX_CPP_VARIANT="{{ E2E_REQUIRED_SPLIT_CPP_VARIANT }}" \
    MODEL_MATRIX_CPP_ENV="{{ E2E_REQUIRED_SPLIT_CPP_ENV }}" \
    MODEL_MATRIX_CPP_WARMUP_RUNS="{{ E2E_REQUIRED_SPLIT_CPP_WARMUP_RUNS }}" \
    MODEL_MATRIX_PYTHON_RESULTS="{{ E2E_REQUIRED_SPLIT_PYTHON_RESULTS }}" \
    just model-matrix "{{ out }}"
    uv run --no-project scripts/summarize_e2e_component_splits.py \
        "{{ out }}/summary.json" \
        --out "{{ out }}/component_split.json"
    uv run --no-project scripts/summarize_e2e_component_splits.py \
        "{{ out }}/summary.json" \
        --markdown \
        --out "{{ out }}/component_split.md"
    uv run --no-project scripts/summarize_e2e_component_splits.py \
        "{{ out }}/summary.json" \
        --process-markdown \
        --out "{{ out }}/process_split.md"
    uv run --no-project scripts/summarize_e2e_component_splits.py \
        "{{ out }}/summary.json" \
        --required-processes \
        --out "{{ out }}/required_process_split.json"
    uv run --no-project scripts/summarize_e2e_component_splits.py \
        "{{ out }}/summary.json" \
        --required-process-markdown \
        --out "{{ out }}/required_process_split.md"

# Profile the same required-E2E C++ contract with synchronized CUDA nodes for attribution.
e2e-required-profile out=E2E_REQUIRED_PROFILE_OUT: build
    mkdir -p "{{ out }}"
    profile_flags=(); \
    [[ -n "{{ E2E_REQUIRED_PROFILE_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" ]] && profile_flags+=(--preencode-cached-tail-frames); \
    [[ -n "{{ E2E_REQUIRED_PROFILE_CPP_TEXT_INIT_SELECTED_ONLY }}" ]] && profile_flags+=(--text-init-selected-only); \
    [[ -n "{{ E2E_REQUIRED_PROFILE_CPP_NO_OUTPUT_ARTIFACTS }}" ]] && profile_flags+=(--no-output-artifacts); \
    env {{ E2E_REQUIRED_PROFILE_CPP_ENV }} SAM3_PROFILE=1 GGML_CUDA_PROFILE_NODES=1 \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir "{{ E2E_REQUIRED_PROFILE_MODELS_DIR }}" \
        --frame-dir "{{ E2E_REQUIRED_PROFILE_FRAMES_DIR }}" \
        --gpu-only --bbox-only --quiet \
        --n-frames "{{ E2E_REQUIRED_PROFILE_FRAMES }}" \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter "{{ E2E_REQUIRED_PROFILE_FILTER }}" \
        --no-isolation --warmup-runs "{{ E2E_REQUIRED_PROFILE_CPP_WARMUP_RUNS }}" \
        --timed-start-frame "{{ E2E_REQUIRED_PROFILE_TIMED_START_FRAME }}" \
        "${profile_flags[@]}" \
        --output-frame-timing-jsonl "{{ out }}/frame_timing.jsonl" \
        > "{{ out }}/benchmark.stdout" 2> "{{ out }}/profile.log"
    uv run --no-project scripts/summarize_cuda_stage_profile.py \
        "{{ out }}/profile.log" \
        --label sam3_encode \
        --out "{{ out }}/stage_profile.json" \
        > "{{ out }}/stage_profile.stdout"
    uv run --no-project scripts/summarize_cuda_node_names.py \
        "{{ out }}/profile.log" \
        --label sam3_encode \
        --split-computes \
        --top 80 \
        --out "{{ out }}/node_names_split.json" \
        > "{{ out }}/node_names_split.stdout"

# Profile GGML CUDA GEMM paths, including cuBLASLt bias-fusion paths, under the same required-E2E C++ contract.
e2e-required-mulmat-profile out=E2E_REQUIRED_MULMAT_PROFILE_OUT: build
    mkdir -p "{{ out }}"
    profile_flags=(); \
    [[ -n "{{ E2E_REQUIRED_PROFILE_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" ]] && profile_flags+=(--preencode-cached-tail-frames); \
    [[ -n "{{ E2E_REQUIRED_PROFILE_CPP_TEXT_INIT_SELECTED_ONLY }}" ]] && profile_flags+=(--text-init-selected-only); \
    [[ -n "{{ E2E_REQUIRED_PROFILE_CPP_NO_OUTPUT_ARTIFACTS }}" ]] && profile_flags+=(--no-output-artifacts); \
    env {{ E2E_REQUIRED_PROFILE_CPP_ENV }} GGML_CUDA_PROFILE_MUL_MAT=1 GGML_CUDA_PROFILE_CUBLASLT_BIAS_FUSION=1 GGML_CUDA_PROFILE_CUBLASLT_BIAS_TIMING=1 \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir "{{ E2E_REQUIRED_PROFILE_MODELS_DIR }}" \
        --frame-dir "{{ E2E_REQUIRED_PROFILE_FRAMES_DIR }}" \
        --gpu-only --bbox-only --quiet \
        --n-frames "{{ E2E_REQUIRED_PROFILE_FRAMES }}" \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter "{{ E2E_REQUIRED_PROFILE_FILTER }}" \
        --no-isolation --warmup-runs "{{ E2E_REQUIRED_PROFILE_CPP_WARMUP_RUNS }}" \
        --timed-start-frame "{{ E2E_REQUIRED_PROFILE_TIMED_START_FRAME }}" \
        "${profile_flags[@]}" \
        --output-frame-timing-jsonl "{{ out }}/frame_timing.jsonl" \
        > "{{ out }}/benchmark.stdout" 2> "{{ out }}/mulmat.log"
    uv run --no-project scripts/summarize_cuda_mulmat_profile.py \
        "{{ out }}/mulmat.log" \
        --out "{{ out }}/mulmat_profile.json" \
        > "{{ out }}/mulmat_profile.stdout"
    uv run --no-project scripts/summarize_cublaslt_bias_timing.py \
        "{{ out }}/mulmat.log" \
        --group-by group \
        > "{{ out }}/cublaslt_bias_by_group.json"
    uv run --no-project scripts/summarize_cublaslt_bias_timing.py \
        "{{ out }}/mulmat.log" \
        --group-by shape \
        > "{{ out }}/cublaslt_bias_by_shape.json"

# Measure the isolated SAM3/SAM3.1 ViT image encoder path that backs the required
# tail image-encode process. Use this as a cheap pre-gate before full E2E A/B.
e2e-required-vit-bench out=E2E_REQUIRED_VIT_BENCH_OUT: (build-target "sam3_vit_batch_bench")
    mkdir -p "{{ out }}"
    bench_flags=(); \
    [[ -n "{{ E2E_REQUIRED_VIT_BENCH_WITH_TRACKER_NECK }}" && -z "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE }}" ]] && bench_flags+=(--with-tracker-neck); \
    [[ -n "{{ E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS }}" ]] && bench_flags+=(--vit-blocks "{{ E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS }}"); \
    [[ -n "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE }}" ]] && bench_flags+=(--block-stage-index "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE_INDEX }}" --block-stage "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE }}"); \
    env {{ E2E_REQUIRED_VIT_BENCH_CPP_ENV }} \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_vit_batch_bench \
        --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
        --image "{{ E2E_REQUIRED_VIT_BENCH_IMAGE }}" \
        --batch-size "{{ E2E_REQUIRED_VIT_BENCH_BATCH_SIZE }}" \
        --warmup-runs "{{ E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS }}" \
        --repeats "{{ E2E_REQUIRED_VIT_BENCH_REPEATS }}" \
        --gpu \
        "${bench_flags[@]}" \
        > "{{ out }}/vit_bench.json" 2> "{{ out }}/vit_bench.stderr"

# Sweep cuBLASLt algorithm indices for the hot SAM3 ViT bias-GEMM shapes.
# These GPU benches intentionally run serially; concurrent runs contend for one
# device and invalidate the isolated required-process timing.
e2e-required-cublaslt-algo-sweep out=E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_OUT: (build-target "sam3_vit_batch_bench")
    mkdir -p "{{ out }}"
    run_bench() { \
        local name="$1"; \
        local env_value="$2"; \
        local bench_out="{{ out }}/$name"; \
        mkdir -p "$bench_out"; \
        bench_flags=(); \
        [[ -n "{{ E2E_REQUIRED_VIT_BENCH_WITH_TRACKER_NECK }}" && -z "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE }}" ]] && bench_flags+=(--with-tracker-neck); \
        [[ -n "{{ E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS }}" ]] && bench_flags+=(--vit-blocks "{{ E2E_REQUIRED_VIT_BENCH_VIT_BLOCKS }}"); \
        [[ -n "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE }}" ]] && bench_flags+=(--block-stage-index "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE_INDEX }}" --block-stage "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE }}"); \
        cmd=(./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_vit_batch_bench \
            --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
            --image "{{ E2E_REQUIRED_VIT_BENCH_IMAGE }}" \
            --batch-size "{{ E2E_REQUIRED_VIT_BENCH_BATCH_SIZE }}" \
            --warmup-runs "{{ E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS }}" \
            --repeats "{{ E2E_REQUIRED_VIT_BENCH_REPEATS }}" \
            --gpu \
            "${bench_flags[@]}"); \
        if [[ -n "$env_value" ]]; then \
            env "$env_value" "${cmd[@]}" > "$bench_out/vit_bench.json" 2> "$bench_out/vit_bench.stderr"; \
        else \
            "${cmd[@]}" > "$bench_out/vit_bench.json" 2> "$bench_out/vit_bench.stderr"; \
        fi; \
    }; \
    run_bench baseline ""; \
    run_bench mlp_fc1_algo1 "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC1=1"; \
    run_bench mlp_fc2_algo1 "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=1"; \
    run_bench mlp_fc2_algo2 "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_MLP_FC2=2"; \
    run_bench qkv_algo1 "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_QKV=1"; \
    run_bench attn_proj_algo1 "GGML_CUDA_CUBLASLT_BIAS_ALGO_INDEX_VIT_ATTN_PROJ=1"; \
    E2E_REQUIRED_AUDIT_COMPONENT_SPLIT="{{ E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_COMPONENT_SPLIT }}" \
    E2E_REQUIRED_AUDIT_NODE_NAMES="{{ E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_NODE_NAMES }}" \
    E2E_REQUIRED_AUDIT_CUBLASLT="{{ E2E_REQUIRED_CUBLASLT_ALGO_SWEEP_CUBLASLT }}" \
    E2E_REQUIRED_AUDIT_ISOLATED_BENCHES="vit/baseline={{ out }}/baseline/vit_bench.json:vit/mlp_fc1_algo1={{ out }}/mlp_fc1_algo1/vit_bench.json:vit/mlp_fc2_algo1={{ out }}/mlp_fc2_algo1/vit_bench.json:vit/mlp_fc2_algo2={{ out }}/mlp_fc2_algo2/vit_bench.json:vit/qkv_algo1={{ out }}/qkv_algo1/vit_bench.json:vit/attn_proj_algo1={{ out }}/attn_proj_algo1/vit_bench.json" \
    just e2e-required-audit "{{ out }}/audit"

# Sweep cumulative block0 stop points to separate the local ViT attention and MLP costs.
e2e-required-vit-stage-sweep out=E2E_REQUIRED_VIT_STAGE_SWEEP_OUT: (build-target "sam3_vit_batch_bench")
    mkdir -p "{{ out }}"
    for stage in {{ E2E_REQUIRED_VIT_STAGE_SWEEP_STAGES }}; do \
        env {{ E2E_REQUIRED_VIT_BENCH_CPP_ENV }} \
            ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_vit_batch_bench \
            --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
            --image "{{ E2E_REQUIRED_VIT_BENCH_IMAGE }}" \
            --batch-size "{{ E2E_REQUIRED_VIT_BENCH_BATCH_SIZE }}" \
            --warmup-runs "{{ E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS }}" \
            --repeats "{{ E2E_REQUIRED_VIT_BENCH_REPEATS }}" \
            --gpu \
            --block-stage-index "{{ E2E_REQUIRED_VIT_BENCH_BLOCK_STAGE_INDEX }}" \
            --block-stage "$stage" \
            > "{{ out }}/stage_${stage}.json" 2> "{{ out }}/stage_${stage}.stderr"; \
    done
    uv run --no-project scripts/summarize_vit_stage_sweep.py "{{ out }}" \
        --out "{{ out }}/summary.json" \
        --markdown-out "{{ out }}/summary.md"

# Capture the real E2E ViT intermediate tensors once, then benchmark each stage
# as an independent graph from its exact stage input. This avoids reading
# cumulative stop-point deltas as local stage costs.
e2e-required-vit-isolated-stage-sweep out=E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_OUT: (build-target "sam3_vit_stage_capture") (build-target "sam31_vit_block_case")
    mkdir -p "{{ out }}"
    env {{ E2E_REQUIRED_VIT_BENCH_CPP_ENV }} \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_vit_stage_capture \
        --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
        --image "{{ E2E_REQUIRED_VIT_BENCH_IMAGE }}" \
        --out-dir "{{ out }}/capture" \
        --block "{{ E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCK }}" \
        --threads 4 \
        --gpu \
        > "{{ out }}/capture_manifest.stdout" 2> "{{ out }}/capture_manifest.stderr"
    env {{ E2E_REQUIRED_VIT_BENCH_CPP_ENV }} \
        uv run --no-project scripts/run_vit_isolated_stage_sweep.py \
        --manifest "{{ out }}/capture/manifest.json" \
        --exe ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_vit_block_case \
        --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
        --block "{{ E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCK }}" \
        --out-dir "{{ out }}/stages" \
        --warmup-runs "{{ E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS }}" \
        --repeats "{{ E2E_REQUIRED_VIT_BENCH_REPEATS }}" \
        --threads 4 \
        --device gpu \
        --stages "{{ E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_STAGES }}"
    uv run --no-project scripts/summarize_vit_stage_sweep.py "{{ out }}/stages" \
        --out "{{ out }}/summary.json" \
        --markdown-out "{{ out }}/summary.md"

# Summarize isolated ViT stage variants against the required-E2E graph budget.
e2e-required-vit-stage-ab out=E2E_REQUIRED_VIT_STAGE_AB_OUT:
    mkdir -p "{{ out }}"
    stage_selector=(); \
    [[ -n "{{ E2E_REQUIRED_VIT_STAGE_AB_STAGE }}" ]] && stage_selector+=(--stage "{{ E2E_REQUIRED_VIT_STAGE_AB_STAGE }}"); \
    [[ -n "{{ E2E_REQUIRED_VIT_STAGE_AB_STAGE_NAME }}" ]] && stage_selector+=(--stage-name "{{ E2E_REQUIRED_VIT_STAGE_AB_STAGE_NAME }}"); \
    uv run --no-project scripts/summarize_vit_stage_variant_ab.py \
        baseline="{{ E2E_REQUIRED_VIT_STAGE_AB_BASELINE }}" \
        candidate="{{ E2E_REQUIRED_VIT_STAGE_AB_CANDIDATE }}" \
        --baseline baseline \
        --drop-initial-samples "{{ E2E_REQUIRED_VIT_STAGE_AB_DROP_INITIAL_SAMPLES }}" \
        --e2e-audit "{{ E2E_REQUIRED_VIT_STAGE_AB_AUDIT }}" \
        --budget-scope "{{ E2E_REQUIRED_VIT_STAGE_AB_BUDGET_SCOPE }}" \
        --out "{{ out }}/stage_ab.json" \
        --markdown-out "{{ out }}/stage_ab.md" \
        "${stage_selector[@]}"

# Sweep cumulative ViT block counts with the same [1024,72,72,1] output contract.
# This is the preferred isolated pre-gate for E2E tail image-encode optimization.
e2e-required-vit-block-sweep out=E2E_REQUIRED_VIT_BLOCK_SWEEP_OUT: (build-target "sam3_vit_batch_bench")
    mkdir -p "{{ out }}"
    for blocks in {{ E2E_REQUIRED_VIT_BLOCK_SWEEP_BLOCKS }}; do \
        env {{ E2E_REQUIRED_VIT_BENCH_CPP_ENV }} \
            ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_vit_batch_bench \
            --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
            --image "{{ E2E_REQUIRED_VIT_BENCH_IMAGE }}" \
            --batch-size "{{ E2E_REQUIRED_VIT_BENCH_BATCH_SIZE }}" \
            --warmup-runs "{{ E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS }}" \
            --repeats "{{ E2E_REQUIRED_VIT_BENCH_REPEATS }}" \
            --gpu \
            --vit-blocks "$blocks" \
            > "{{ out }}/blocks_${blocks}.json" 2> "{{ out }}/blocks_${blocks}.stderr"; \
    done
    if [[ -n "{{ E2E_REQUIRED_VIT_BLOCK_SWEEP_WITH_NECK }}" ]]; then \
        env {{ E2E_REQUIRED_VIT_BENCH_CPP_ENV }} \
            ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_vit_batch_bench \
            --model "{{ E2E_REQUIRED_VIT_BENCH_MODEL }}" \
            --image "{{ E2E_REQUIRED_VIT_BENCH_IMAGE }}" \
            --batch-size "{{ E2E_REQUIRED_VIT_BENCH_BATCH_SIZE }}" \
            --warmup-runs "{{ E2E_REQUIRED_VIT_BENCH_WARMUP_RUNS }}" \
            --repeats "{{ E2E_REQUIRED_VIT_BENCH_REPEATS }}" \
            --gpu \
            --with-tracker-neck \
            > "{{ out }}/full_with_tracker_neck.json" 2> "{{ out }}/full_with_tracker_neck.stderr"; \
    fi
    uv run --no-project scripts/summarize_vit_block_sweep.py "{{ out }}" \
        --out "{{ out }}/summary.json" \
        --markdown-out "{{ out }}/summary.md"

# Run the complete same-contract required-E2E measurement bundle for one state.
e2e-required-measure out=E2E_REQUIRED_MEASURE_OUT:
    mkdir -p "{{ out }}"
    just e2e-required-split "{{ out }}/split"
    E2E_REQUIRED_PROFILE_FRAMES_DIR="{{ out }}/split/frames" \
    E2E_REQUIRED_PROFILE_FRAMES="{{ E2E_REQUIRED_SPLIT_FRAMES }}" \
    just e2e-required-profile "{{ out }}/profile"
    E2E_REQUIRED_PROFILE_FRAMES_DIR="{{ out }}/split/frames" \
    E2E_REQUIRED_PROFILE_FRAMES="{{ E2E_REQUIRED_SPLIT_FRAMES }}" \
    just e2e-required-mulmat-profile "{{ out }}/mulmat-profile"
    E2E_REQUIRED_VIT_BENCH_IMAGE="{{ out }}/split/frames/00001.jpg" \
    just e2e-required-vit-bench "{{ out }}/vit-bench"
    if [[ -n "{{ E2E_REQUIRED_MEASURE_ISOLATED_STAGE_SWEEP }}" ]]; then \
        for block in {{ E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCKS }}; do \
            E2E_REQUIRED_VIT_BENCH_IMAGE="{{ out }}/split/frames/00001.jpg" \
            E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCK="$block" \
            just e2e-required-vit-isolated-stage-sweep "{{ out }}/vit-isolated-stage-sweep-block-${block}"; \
        done; \
    fi
    E2E_REQUIRED_PYTHON_PARITY_FRAMES_DIR="{{ out }}/split/frames" \
    E2E_REQUIRED_PYTHON_PARITY_FRAMES="{{ E2E_REQUIRED_SPLIT_FRAMES }}" \
    E2E_REQUIRED_PYTHON_PARITY_MODELS_DIR="{{ E2E_REQUIRED_SPLIT_MODELS_DIR }}" \
    E2E_REQUIRED_PYTHON_PARITY_FILTER="{{ E2E_REQUIRED_SPLIT_FILTER }}" \
    E2E_REQUIRED_PYTHON_PARITY_CPP_WARMUP_RUNS="{{ E2E_REQUIRED_SPLIT_CPP_WARMUP_RUNS }}" \
    E2E_REQUIRED_PYTHON_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES="{{ E2E_REQUIRED_SPLIT_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" \
    E2E_REQUIRED_PYTHON_PARITY_CPP_TEXT_INIT_SELECTED_ONLY="{{ E2E_REQUIRED_SPLIT_CPP_TEXT_INIT_SELECTED_ONLY }}" \
    E2E_REQUIRED_PYTHON_PARITY_DTYPE="{{ E2E_REQUIRED_SPLIT_PYTHON_DTYPE }}" \
    E2E_REQUIRED_PYTHON_PARITY_TF32_POLICY="{{ E2E_REQUIRED_SPLIT_TF32_POLICY }}" \
    E2E_REQUIRED_PYTHON_PARITY_VERSION="{{ E2E_REQUIRED_SPLIT_PYTHON_SAM3_VERSION }}" \
    just e2e-required-python-parity "{{ out }}/python-parity"
    isolated_benches="tail-vit={{ out }}/vit-bench/vit_bench.json"; \
    if [[ -n "{{ E2E_REQUIRED_MEASURE_ISOLATED_STAGE_SWEEP }}" ]]; then \
        for block in {{ E2E_REQUIRED_VIT_ISOLATED_STAGE_SWEEP_BLOCKS }}; do \
            summary="{{ out }}/vit-isolated-stage-sweep-block-${block}/summary.json"; \
            if [[ -f "$summary" ]]; then \
                isolated_benches="$isolated_benches:tail-stage-block${block}=$summary"; \
            fi; \
        done; \
    fi; \
    E2E_REQUIRED_AUDIT_COMPONENT_SPLIT="{{ out }}/split/component_split.json" \
    E2E_REQUIRED_AUDIT_NODE_NAMES="{{ out }}/profile/node_names_split.json" \
    E2E_REQUIRED_AUDIT_STAGE_PROFILE="{{ out }}/profile/stage_profile.json" \
    E2E_REQUIRED_AUDIT_CUBLASLT="{{ out }}/mulmat-profile/cublaslt_bias_by_group.json" \
    E2E_REQUIRED_AUDIT_PYTHON_PARITY="official-python={{ out }}/python-parity/python-vs-cpp.json" \
    E2E_REQUIRED_AUDIT_ISOLATED_BENCHES="$isolated_benches" \
        just e2e-required-audit "{{ out }}/audit"

# Run the complete required-E2E measurement bundle for SAM3 F16 against official Python fp16/TF32.
e2e-required-measure-sam3-f16 out=E2E_REQUIRED_MEASURE_OUT:
    E2E_REQUIRED_SPLIT_FILTER="sam3-f16" \
    E2E_REQUIRED_SPLIT_PYTHON_DTYPE="fp16" \
    E2E_REQUIRED_PROFILE_FILTER="sam3-f16" \
    E2E_REQUIRED_MULMAT_PROFILE_OUT="{{ out }}/mulmat-profile" \
    E2E_REQUIRED_VIT_BENCH_MODEL="models/sam3/sam3-f16.ggml" \
    just e2e-required-measure "{{ out }}"

# Combine required-E2E process, CUDA-node, and cuBLASLt attribution into one target ranking.
e2e-required-audit out=E2E_REQUIRED_AUDIT_OUT:
    mkdir -p "{{ out }}"
    candidate_args=(); \
    if [[ -n "{{ E2E_REQUIRED_AUDIT_CANDIDATE_COMPONENT_SPLITS }}" ]]; then \
        IFS=':' read -ra candidate_splits <<< "{{ E2E_REQUIRED_AUDIT_CANDIDATE_COMPONENT_SPLITS }}"; \
        for split in "${candidate_splits[@]}"; do \
            [[ -n "$split" ]] && candidate_args+=(--candidate-component-split "$split"); \
        done; \
    fi; \
    parity_args=(); \
    if [[ -n "{{ E2E_REQUIRED_AUDIT_CANDIDATE_PARITY }}" ]]; then \
        IFS=':' read -ra candidate_parity <<< "{{ E2E_REQUIRED_AUDIT_CANDIDATE_PARITY }}"; \
        for parity in "${candidate_parity[@]}"; do \
            [[ -n "$parity" ]] && parity_args+=(--candidate-parity "$parity"); \
        done; \
    fi; \
    python_parity_args=(); \
    if [[ -n "{{ E2E_REQUIRED_AUDIT_PYTHON_PARITY }}" ]]; then \
        IFS=':' read -ra python_parity <<< "{{ E2E_REQUIRED_AUDIT_PYTHON_PARITY }}"; \
        for parity in "${python_parity[@]}"; do \
            [[ -n "$parity" ]] && python_parity_args+=(--python-parity "$parity"); \
        done; \
    fi; \
    isolated_args=(); \
    if [[ -n "{{ E2E_REQUIRED_AUDIT_ISOLATED_BENCHES }}" ]]; then \
        IFS=':' read -ra isolated_benches <<< "{{ E2E_REQUIRED_AUDIT_ISOLATED_BENCHES }}"; \
        for bench in "${isolated_benches[@]}"; do \
            [[ -n "$bench" ]] && isolated_args+=(--isolated-bench "$bench"); \
        done; \
    fi; \
    fattn_args=(); \
    if [[ -n "{{ E2E_REQUIRED_AUDIT_FATTN_PROFILES }}" ]]; then \
        IFS=':' read -ra fattn_profiles <<< "{{ E2E_REQUIRED_AUDIT_FATTN_PROFILES }}"; \
        for profile in "${fattn_profiles[@]}"; do \
            [[ -n "$profile" ]] && fattn_args+=(--fattn-profile "$profile"); \
        done; \
    fi; \
    stage_profile_args=(); \
    if [[ -n "{{ E2E_REQUIRED_AUDIT_STAGE_PROFILE }}" ]]; then \
        stage_profile_args+=(--stage-profile "{{ E2E_REQUIRED_AUDIT_STAGE_PROFILE }}"); \
    fi; \
    uv run --no-project scripts/summarize_e2e_optimization_targets.py \
        --component-split "{{ E2E_REQUIRED_AUDIT_COMPONENT_SPLIT }}" \
        --node-names "{{ E2E_REQUIRED_AUDIT_NODE_NAMES }}" \
        --node-label "{{ E2E_REQUIRED_AUDIT_NODE_LABEL }}" \
        --cublaslt "{{ E2E_REQUIRED_AUDIT_CUBLASLT }}" \
        --out "{{ out }}/optimization_targets.json" \
        --markdown-out "{{ out }}/optimization_targets.md" \
        "${stage_profile_args[@]}" \
        "${candidate_args[@]}" \
        "${parity_args[@]}" \
        "${python_parity_args[@]}" \
        "${isolated_args[@]}" \
        "${fattn_args[@]}"

# Run artifact-producing baseline/candidate checks under the required-E2E contract.
e2e-required-parity out=E2E_REQUIRED_PARITY_OUT: (build-target "sam3_benchmark")
    mkdir -p "{{ out }}/baseline-masks" "{{ out }}/candidate-masks"
    parity_flags=(); \
    [[ -z "{{ E2E_REQUIRED_PARITY_CPP_MASK_OUTPUT }}" ]] && parity_flags+=(--bbox-only); \
    [[ -n "{{ E2E_REQUIRED_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" ]] && parity_flags+=(--preencode-cached-tail-frames); \
    [[ -n "{{ E2E_REQUIRED_PARITY_CPP_TEXT_INIT_SELECTED_ONLY }}" ]] && parity_flags+=(--text-init-selected-only); \
    env {{ E2E_REQUIRED_PARITY_BASELINE_CPP_ENV }} \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir "{{ E2E_REQUIRED_PARITY_MODELS_DIR }}" \
        --frame-dir "{{ E2E_REQUIRED_PARITY_FRAMES_DIR }}" \
        --gpu-only --quiet \
        --n-frames "{{ E2E_REQUIRED_PARITY_FRAMES }}" \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter "{{ E2E_REQUIRED_PARITY_FILTER }}" \
        --warmup-runs "{{ E2E_REQUIRED_PARITY_CPP_WARMUP_RUNS }}" \
        --timed-start-frame "{{ E2E_REQUIRED_PARITY_TIMED_START_FRAME }}" \
        "${parity_flags[@]}" \
        --output-jsonl "{{ out }}/baseline.jsonl" \
        --output-mask-dir "{{ out }}/baseline-masks" \
        > "{{ out }}/baseline.stdout" 2> "{{ out }}/baseline.stderr"
    parity_flags=(); \
    [[ -z "{{ E2E_REQUIRED_PARITY_CPP_MASK_OUTPUT }}" ]] && parity_flags+=(--bbox-only); \
    [[ -n "{{ E2E_REQUIRED_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" ]] && parity_flags+=(--preencode-cached-tail-frames); \
    [[ -n "{{ E2E_REQUIRED_PARITY_CPP_TEXT_INIT_SELECTED_ONLY }}" ]] && parity_flags+=(--text-init-selected-only); \
    env {{ E2E_REQUIRED_PARITY_CANDIDATE_CPP_ENV }} \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir "{{ E2E_REQUIRED_PARITY_MODELS_DIR }}" \
        --frame-dir "{{ E2E_REQUIRED_PARITY_FRAMES_DIR }}" \
        --gpu-only --quiet \
        --n-frames "{{ E2E_REQUIRED_PARITY_FRAMES }}" \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter "{{ E2E_REQUIRED_PARITY_FILTER }}" \
        --warmup-runs "{{ E2E_REQUIRED_PARITY_CPP_WARMUP_RUNS }}" \
        --timed-start-frame "{{ E2E_REQUIRED_PARITY_TIMED_START_FRAME }}" \
        "${parity_flags[@]}" \
        --output-jsonl "{{ out }}/candidate.jsonl" \
        --output-mask-dir "{{ out }}/candidate-masks" \
        > "{{ out }}/candidate.stdout" 2> "{{ out }}/candidate.stderr"
    uv run --no-project scripts/compare_tracking_jsonl.py \
        "{{ out }}/baseline.jsonl" \
        "{{ out }}/candidate.jsonl" \
        --lhs-label "{{ E2E_REQUIRED_PARITY_BASELINE_VARIANT }}" \
        --rhs-label "{{ E2E_REQUIRED_PARITY_CANDIDATE_VARIANT }}" \
        --out "{{ out }}/compare.json" \
        > "{{ out }}/compare.stdout"

# Compare official Python artifacts with the C++ required-E2E contract artifacts.
# This is a quality gate for final Python parity; it intentionally allows small
# mask/hash differences while reporting exact deltas.
e2e-required-python-parity out=E2E_REQUIRED_PYTHON_PARITY_OUT: (build-target "sam3_benchmark")
    mkdir -p "{{ out }}/python-masks" "{{ out }}/cpp-masks"
    uv run --no-project scripts/dump_sam3_python_tracking_jsonl.py \
        --sam3-repo "{{ SAM3_REPO }}" \
        --frame-dir "{{ E2E_REQUIRED_PYTHON_PARITY_FRAMES_DIR }}" \
        --out-jsonl "{{ out }}/python.jsonl" \
        --out-candidates-jsonl "{{ out }}/python-candidates.jsonl" \
        --out-mask-dir "{{ out }}/python-masks" \
        --frames "{{ E2E_REQUIRED_PYTHON_PARITY_FRAMES }}" \
        --prompt person \
        --dtype "{{ E2E_REQUIRED_PYTHON_PARITY_DTYPE }}" \
        --tf32 "{{ E2E_REQUIRED_PYTHON_PARITY_TF32_POLICY }}" \
        --version "{{ E2E_REQUIRED_PYTHON_PARITY_VERSION }}" \
        > "{{ out }}/python.stdout" 2> "{{ out }}/python.stderr"
    cpp_flags=(); \
    [[ -n "{{ E2E_REQUIRED_PYTHON_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" ]] && cpp_flags+=(--preencode-cached-tail-frames); \
    [[ -n "{{ E2E_REQUIRED_PYTHON_PARITY_CPP_TEXT_INIT_SELECTED_ONLY }}" ]] && cpp_flags+=(--text-init-selected-only); \
    env {{ E2E_REQUIRED_PYTHON_PARITY_CPP_ENV }} \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir "{{ E2E_REQUIRED_PYTHON_PARITY_MODELS_DIR }}" \
        --frame-dir "{{ E2E_REQUIRED_PYTHON_PARITY_FRAMES_DIR }}" \
        --gpu-only --quiet \
        --n-frames "{{ E2E_REQUIRED_PYTHON_PARITY_FRAMES }}" \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter "{{ E2E_REQUIRED_PYTHON_PARITY_FILTER }}" \
        --warmup-runs "{{ E2E_REQUIRED_PYTHON_PARITY_CPP_WARMUP_RUNS }}" \
        --timed-start-frame "{{ E2E_REQUIRED_PYTHON_PARITY_TIMED_START_FRAME }}" \
        "${cpp_flags[@]}" \
        --output-jsonl "{{ out }}/cpp.jsonl" \
        --output-mask-dir "{{ out }}/cpp-masks" \
        > "{{ out }}/cpp.stdout" 2> "{{ out }}/cpp.stderr"
    uv run --no-project scripts/compare_tracking_jsonl.py \
        "{{ out }}/python.jsonl" \
        "{{ out }}/cpp.jsonl" \
        --lhs-label "official-python-{{ E2E_REQUIRED_PYTHON_PARITY_DTYPE }}" \
        --rhs-label "sam3cpp-{{ E2E_REQUIRED_PYTHON_PARITY_FILTER }}" \
        --min-bbox-iou 0.9 \
        --max-bbox-delta-px 20 \
        --max-score-delta 0.2 \
        --max-mask-area-rel-delta 0.2 \
        --min-mask-iou 0.95 \
        --allow-mask-hash-diff \
        --out "{{ out }}/python-vs-cpp.json" \
        > "{{ out }}/python-vs-cpp.stdout"

# Run a baseline/candidate required-E2E A/B under one input contract, then compare the split.
e2e-required-ab out=E2E_REQUIRED_AB_OUT:
    mkdir -p "{{ out }}"
    E2E_REQUIRED_SPLIT_CPP_VARIANT="{{ E2E_REQUIRED_AB_BASELINE_VARIANT }}" \
    E2E_REQUIRED_SPLIT_CPP_ENV="{{ E2E_REQUIRED_AB_BASELINE_CPP_ENV }}" \
    E2E_REQUIRED_SPLIT_PYTHON_RESULTS="" \
    just e2e-required-split "{{ out }}/baseline"
    E2E_REQUIRED_SPLIT_CPP_VARIANT="{{ E2E_REQUIRED_AB_CANDIDATE_VARIANT }}" \
    E2E_REQUIRED_SPLIT_CPP_ENV="{{ E2E_REQUIRED_AB_CANDIDATE_CPP_ENV }}" \
    E2E_REQUIRED_SPLIT_PYTHON_RESULTS="{{ out }}/baseline/python-results.json" \
    just e2e-required-split "{{ out }}/candidate"
    E2E_REQUIRED_PARITY_FRAMES_DIR="{{ out }}/baseline/frames" \
    E2E_REQUIRED_PARITY_FRAMES="{{ E2E_REQUIRED_SPLIT_FRAMES }}" \
    E2E_REQUIRED_PARITY_MODELS_DIR="{{ E2E_REQUIRED_SPLIT_MODELS_DIR }}" \
    E2E_REQUIRED_PARITY_FILTER="{{ E2E_REQUIRED_SPLIT_FILTER }}" \
    E2E_REQUIRED_PARITY_BASELINE_VARIANT="{{ E2E_REQUIRED_AB_BASELINE_VARIANT }}" \
    E2E_REQUIRED_PARITY_BASELINE_CPP_ENV="{{ E2E_REQUIRED_AB_BASELINE_CPP_ENV }}" \
    E2E_REQUIRED_PARITY_CANDIDATE_VARIANT="{{ E2E_REQUIRED_AB_CANDIDATE_VARIANT }}" \
    E2E_REQUIRED_PARITY_CANDIDATE_CPP_ENV="{{ E2E_REQUIRED_AB_CANDIDATE_CPP_ENV }}" \
    E2E_REQUIRED_PARITY_CPP_WARMUP_RUNS="{{ E2E_REQUIRED_SPLIT_CPP_WARMUP_RUNS }}" \
    E2E_REQUIRED_PARITY_CPP_PREENCODE_CACHED_TAIL_FRAMES="{{ E2E_REQUIRED_SPLIT_CPP_PREENCODE_CACHED_TAIL_FRAMES }}" \
    E2E_REQUIRED_PARITY_CPP_TEXT_INIT_SELECTED_ONLY="{{ E2E_REQUIRED_SPLIT_CPP_TEXT_INIT_SELECTED_ONLY }}" \
    E2E_REQUIRED_PARITY_CPP_MASK_OUTPUT="{{ E2E_REQUIRED_SPLIT_CPP_MASK_OUTPUT }}" \
    just e2e-required-parity "{{ out }}/parity"
    uv run --no-project scripts/summarize_e2e_optimization_targets.py \
        --component-split "{{ out }}/baseline/component_split.json" \
        --candidate-component-split "{{ out }}/candidate/component_split.json" \
        --candidate-parity "{{ E2E_REQUIRED_AB_CANDIDATE_VARIANT }}={{ out }}/parity/compare.json" \
        --out "{{ out }}/optimization_targets.json" \
        --markdown-out "{{ out }}/optimization_targets.md"

# Run the required-E2E split for SAM3 F16 against official Python fp16/TF32.
e2e-required-split-sam3-f16 out=E2E_REQUIRED_SPLIT_OUT:
    E2E_REQUIRED_SPLIT_FILTER="sam3-f16" \
    E2E_REQUIRED_SPLIT_PYTHON_DTYPE="fp16" \
    just e2e-required-split "{{ out }}"

# Run a SAM3 F16 required-E2E A/B with matching official Python fp16/TF32.
e2e-required-ab-sam3-f16 out=E2E_REQUIRED_AB_OUT:
    E2E_REQUIRED_SPLIT_FILTER="sam3-f16" \
    E2E_REQUIRED_PARITY_FILTER="sam3-f16" \
    E2E_REQUIRED_SPLIT_PYTHON_DTYPE="fp16" \
    just e2e-required-ab "{{ out }}"

# Measure cuDNN SDPA against the two SAM3 ViT head64 attention shapes and a fresh ggml FATTN profile.
cudnn-sdpa-head64 out=CUDNN_SDPA_OUT: build (build-target "sam3_cudnn_sdpa_bench")
    mkdir -p {{ out }}
    GGML_CUDA_PROFILE_FATTN=1 ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir {{ MODELS_DIR }} \
        --video {{ VIDEO }} \
        --gpu-only --bbox-only --quiet --n-frames 3 \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter {{ CUDNN_SDPA_MODEL_FILTER }} --no-isolation \
        > {{ out }}/fattn_run.log 2> {{ out }}/fattn.log
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_cudnn_sdpa_bench \
        --batch 1 --heads 16 --seq 5184 --head-dim 64 --warmup 5 --iters 30 \
        > {{ out }}/global_5184.log
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_cudnn_sdpa_bench \
        --batch 9 --heads 16 --seq 576 --head-dim 64 --warmup 5 --iters 60 \
        > {{ out }}/window_576x9.log
    uv run --no-project scripts/summarize_cudnn_sdpa_gap.py \
        --fattn {{ out }}/fattn.log \
        --cudnn {{ out }}/global_5184.log {{ out }}/window_576x9.log \
        --drop-first-per-shape \
        --out {{ out }}/summary.json

# Capture CUDA device visibility and strict GPU benchmark readiness.
cuda-health out=CUDA_HEALTH_OUT: build
    mkdir -p {{ out }}
    { \
        date -Is; \
        nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu --format=csv,noheader; \
    } > {{ out }}/nvidia-smi.log 2>&1 || true
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir models/sam3 \
        --video {{ VIDEO }} \
        --gpu-only --bbox-only --quiet --n-frames 2 \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter sam3-bf16 --no-isolation \
        > {{ out }}/sam3-bf16-gpu-smoke.log 2>&1 || true
    if rg -q "SUMMARY: 1 runs, 1 OK, 0 FAIL" {{ out }}/sam3-bf16-gpu-smoke.log; then \
        printf 'cuda_health=ok\n' > {{ out }}/summary.env; \
    else \
        printf 'cuda_health=blocked\n' > {{ out }}/summary.env; \
    fi
    cat {{ out }}/summary.env

# Regenerate SAM3.1 checkpoint inventory and renamed-key coverage against SAM3.
sam31-contract out=SAM31_CONTRACT_OUT:
    mkdir -p {{ out }}
    uv run scripts/summarize_sam31_checkpoint_contract.py \
        --sam3 {{ SAM31_CONTRACT_SAM3 }} \
        --sam31 {{ SAM31_CONTRACT_SAM31 }} \
        --inventory-out {{ out }}/inventory.json \
        --coverage-out {{ out }}/sam3-vs-sam31-renamed-coverage.json
    uv run scripts/check_sam31_contract.py \
        --inventory {{ out }}/inventory.json \
        --coverage {{ out }}/sam3-vs-sam31-renamed-coverage.json \
        --cxx-header sam3_sam31.h \
        --cxx-tensor-manifest sam3_sam31_tensor_manifest.h \
        --cxx-source sam3.cpp \
        --cxx-api sam3.h \
        --converter convert_sam3_to_ggml.py

# Generate a C++ tensor-name table for one SAM3.1 implementation manifest slice.
sam31-slice-header slice=SAM31_SLICE out=SAM31_CONTRACT_OUT: sam31-contract
    uv run scripts/generate_sam31_slice_tensor_header.py \
        --coverage {{ out }}/sam3-vs-sam31-renamed-coverage.json \
        --slice {{ slice }} \
        --out {{ out }}/{{ slice }}_tensor_names.h
    {{ CXX_COMPILER }} -std=c++23 -x c++ -fsyntax-only {{ out }}/{{ slice }}_tensor_names.h

# Generate and syntax-check C++ tensor-name tables for every SAM3.1 manifest slice.
sam31-slice-headers-all out=SAM31_CONTRACT_OUT: sam31-contract
    for slice in multiplex_mask_decoder multiplex_memory_backbone multiplex_tracker_transformer multiplex_detector_feature_adapters; do \
        uv run scripts/generate_sam31_slice_tensor_header.py \
            --coverage {{ out }}/sam3-vs-sam31-renamed-coverage.json \
            --slice "$slice" \
            --out {{ out }}/${slice}_tensor_names.h; \
        {{ CXX_COMPILER }} -std=c++23 -x c++ -fsyntax-only {{ out }}/${slice}_tensor_names.h; \
    done

# Validate the C++ SAM3.1 Object Multiplex bucket/mux/demux contract.
sam31-multiplex-state: (build-target "sam31_multiplex_state")
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_multiplex_state

# Compare the C++ SAM3.1 multiplex helper against official Python SAM3.
sam31-multiplex-parity out=SAM31_MULTIPLEX_STATE_OUT: (build-target "sam31_multiplex_state")
    mkdir -p {{ out }}
    uv run scripts/compare_sam31_multiplex_state.py \
        --repo {{ SAM3_REPO }} \
        --cpp-exe ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_multiplex_state \
        --out {{ out }}/summary.json

# Dump an official Python SAM3.1 multiplex mask-decoder case for C++ graph parity.
sam31-mux-mask-decoder-case out=SAM31_MUX_MASK_DECODER_OUT:
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/dump_sam31_mux_mask_decoder_case.py \
        --repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --feat-size {{ SAM31_MUX_MASK_DECODER_FEAT_SIZE }} \
        --buckets {{ SAM31_MUX_MASK_DECODER_BUCKETS }}

# Convert the SAM3.1 checkpoint into the local ggml container used by the C++ slice runner.
sam31-convert-ggml checkpoint=SAM31_CONTRACT_SAM31 out=SAM31_MUX_MASK_DECODER_MODEL:
    mkdir -p "$(dirname "{{ out }}")"
    uv run convert_sam3_to_ggml.py --model {{ checkpoint }} --output {{ out }} --ftype 0 --sam31 --sam31-mask-decoder-only

# Run the C++ SAM3.1 multiplex mask-decoder slice and compare it with the Python reference.
sam31-mux-mask-decoder-parity out=SAM31_MUX_MASK_DECODER_COMPARE_OUT: sam31-mux-mask-decoder-case (build-target "sam31_mux_mask_decoder_case")
    if [ ! -s "{{ SAM31_MUX_MASK_DECODER_MODEL }}" ]; then just sam31-convert-ggml; fi
    mkdir -p {{ SAM31_MUX_MASK_DECODER_CPP_OUT }} {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_mux_mask_decoder_case \
        --model {{ SAM31_MUX_MASK_DECODER_MODEL }} \
        --case {{ SAM31_MUX_MASK_DECODER_OUT }} \
        --out {{ SAM31_MUX_MASK_DECODER_CPP_OUT }} \
        --threads 4 \
        --cpu
    uv run scripts/compare_sam31_mux_mask_decoder_case.py \
        --expected {{ SAM31_MUX_MASK_DECODER_OUT }}/expected_cpp_layout \
        --actual {{ SAM31_MUX_MASK_DECODER_CPP_OUT }} \
        --out {{ out }}/summary.json

# Dump an official Python SAM3.1 multiplex memory-backbone case for C++ graph parity.
sam31-memory-backbone-case out=SAM31_MEMORY_BACKBONE_OUT:
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/dump_sam31_memory_backbone_case.py \
        --repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --feat-size {{ SAM31_MEMORY_BACKBONE_FEAT_SIZE }} \
        --mask-size {{ SAM31_MEMORY_BACKBONE_MASK_SIZE }} \
        --buckets {{ SAM31_MEMORY_BACKBONE_BUCKETS }}

# Convert only the SAM3.1 memory backbone into the local ggml container used by the C++ slice runner.
sam31-convert-memory-backbone-ggml checkpoint=SAM31_CONTRACT_SAM31 out=SAM31_MEMORY_BACKBONE_MODEL:
    mkdir -p "$(dirname "{{ out }}")"
    uv run convert_sam3_to_ggml.py --model {{ checkpoint }} --output {{ out }} --ftype 0 --sam31 --sam31-memory-backbone-only

# Run the C++ SAM3.1 memory-backbone slice and compare it with the Python reference.
sam31-memory-backbone-parity out=SAM31_MEMORY_BACKBONE_COMPARE_OUT: sam31-memory-backbone-case (build-target "sam31_memory_backbone_case")
    just sam31-convert-memory-backbone-ggml
    mkdir -p {{ SAM31_MEMORY_BACKBONE_CPP_OUT }} {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_memory_backbone_case \
        --model {{ SAM31_MEMORY_BACKBONE_MODEL }} \
        --case {{ SAM31_MEMORY_BACKBONE_OUT }} \
        --out {{ SAM31_MEMORY_BACKBONE_CPP_OUT }} \
        --threads 4 \
        --{{ SAM31_MEMORY_BACKBONE_DEVICE }}
    uv run scripts/compare_raw_tensor_dirs.py \
        --expected {{ SAM31_MEMORY_BACKBONE_OUT }}/expected_cpp_layout \
        --actual {{ SAM31_MEMORY_BACKBONE_CPP_OUT }} \
        --tensor mask_downsampled \
        --tensor pix_feat_proj \
        --tensor fused_input \
        --tensor fuser0 \
        --tensor vision_features \
        --tensor vision_pos_enc \
        --tensor stored_spatial_feats \
        --tensor stored_spatial_pe \
        --tensor memory_prompt \
        --tensor memory_prompt_pos \
        --max-abs-tolerance {{ SAM31_MEMORY_BACKBONE_MAX_ABS_TOLERANCE }} \
        --mean-abs-tolerance {{ SAM31_MEMORY_BACKBONE_MEAN_ABS_TOLERANCE }} \
        --out {{ out }}/summary.json

# Dump an official-weight SAM3.1 propagation feature-adapter case for C++ graph parity.
sam31-propagation-features-case out=SAM31_PROPAGATION_FEATURES_OUT:
    uv run scripts/dump_sam31_propagation_features_case.py \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --feat-size {{ SAM31_PROPAGATION_FEATURES_FEAT_SIZE }}

# Convert only the SAM3.1 propagation feature adapter and high-res decoder projections.
sam31-convert-propagation-features-ggml checkpoint=SAM31_CONTRACT_SAM31 out=SAM31_PROPAGATION_FEATURES_MODEL:
    mkdir -p "$(dirname "{{ out }}")"
    uv run convert_sam3_to_ggml.py --model {{ checkpoint }} --output {{ out }} --ftype 0 --sam31 --sam31-propagation-features-only

# Run the C++ SAM3.1 propagation feature-adapter slice and compare it with the Python reference.
sam31-propagation-features-parity out=SAM31_PROPAGATION_FEATURES_COMPARE_OUT: sam31-propagation-features-case (build-target "sam31_propagation_features_case")
    just sam31-convert-propagation-features-ggml
    mkdir -p {{ SAM31_PROPAGATION_FEATURES_CPP_OUT }} {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_propagation_features_case \
        --model {{ SAM31_PROPAGATION_FEATURES_MODEL }} \
        --case {{ SAM31_PROPAGATION_FEATURES_OUT }} \
        --out {{ SAM31_PROPAGATION_FEATURES_CPP_OUT }} \
        --threads 4 \
        --{{ SAM31_PROPAGATION_FEATURES_DEVICE }}
    uv run scripts/compare_raw_tensor_dirs.py \
        --expected {{ SAM31_PROPAGATION_FEATURES_OUT }}/expected_cpp_layout \
        --actual {{ SAM31_PROPAGATION_FEATURES_CPP_OUT }} \
        --tensor projected_s0 \
        --tensor projected_s1 \
        --tensor image_features \
        --max-abs-tolerance {{ SAM31_PROPAGATION_FEATURES_MAX_ABS_TOLERANCE }} \
        --mean-abs-tolerance {{ SAM31_PROPAGATION_FEATURES_MEAN_ABS_TOLERANCE }} \
        --out {{ out }}/summary.json

sam31-memory-attention-case out=SAM31_MEMORY_ATTENTION_OUT:
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/dump_sam31_memory_attention_case.py \
        --repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --feat-size {{ SAM31_MEMORY_ATTENTION_FEAT_SIZE }} \
        --buckets {{ SAM31_MEMORY_ATTENTION_BUCKETS }} \
        --dtype {{ SAM31_MEMORY_ATTENTION_DTYPE }} \
        --device {{ SAM31_MEMORY_ATTENTION_PY_DEVICE }}

# Convert only the SAM3.1 memory-attention encoder into the local ggml container used by the C++ slice runner.
sam31-convert-memory-attention-ggml checkpoint=SAM31_CONTRACT_SAM31 out=SAM31_MEMORY_ATTENTION_MODEL:
    mkdir -p "$(dirname "{{ out }}")"
    uv run convert_sam3_to_ggml.py --model {{ checkpoint }} --output {{ out }} --ftype {{ SAM31_MEMORY_ATTENTION_FTYPE }} --sam31 --sam31-memory-attention-only

# Run the C++ SAM3.1 memory-attention slice and compare it with the Python reference.
sam31-memory-attention-parity out=SAM31_MEMORY_ATTENTION_COMPARE_OUT: sam31-memory-attention-case (build-target "sam31_memory_attention_case")
    just sam31-convert-memory-attention-ggml
    mkdir -p {{ SAM31_MEMORY_ATTENTION_CPP_OUT }} {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_memory_attention_case \
        --model {{ SAM31_MEMORY_ATTENTION_MODEL }} \
        --case {{ SAM31_MEMORY_ATTENTION_OUT }} \
        --out {{ SAM31_MEMORY_ATTENTION_CPP_OUT }} \
        --threads 4 \
        --{{ SAM31_MEMORY_ATTENTION_DEVICE }}
    uv run scripts/compare_raw_tensor_dirs.py \
        --expected {{ SAM31_MEMORY_ATTENTION_OUT }}/expected_cpp_layout \
        --actual {{ SAM31_MEMORY_ATTENTION_CPP_OUT }} \
        --tensor sam31_mem_attn_input \
        --tensor sam31_mem_attn_layer0_after_sa \
        --tensor sam31_mem_attn_layer0_after_ca \
        --tensor sam31_mem_attn_layer0_after_ffn \
        --tensor sam31_mem_attn_layer1_after_sa \
        --tensor sam31_mem_attn_layer1_after_ca \
        --tensor sam31_mem_attn_layer1_after_ffn \
        --tensor sam31_mem_attn_layer2_after_sa \
        --tensor sam31_mem_attn_layer2_after_ca \
        --tensor sam31_mem_attn_layer2_after_ffn \
        --tensor sam31_mem_attn_layer3_after_sa \
        --tensor sam31_mem_attn_layer3_after_ca \
        --tensor sam31_mem_attn_layer3_after_ffn \
        --tensor sam31_mem_attn_output \
        --max-abs-tolerance {{ SAM31_MEMORY_ATTENTION_MAX_ABS_TOLERANCE }} \
        --mean-abs-tolerance {{ SAM31_MEMORY_ATTENTION_MEAN_ABS_TOLERANCE }} \
        --out {{ out }}/summary.json

sam31-memory-attention-decoder-case out=SAM31_MEMORY_ATTENTION_DECODER_OUT:
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/dump_sam31_memory_attention_decoder_case.py \
        --repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --feat-size {{ SAM31_MEMORY_ATTENTION_DECODER_FEAT_SIZE }} \
        --buckets {{ SAM31_MEMORY_ATTENTION_DECODER_BUCKETS }} \
        --dtype {{ SAM31_MEMORY_ATTENTION_DECODER_DTYPE }} \
        --device {{ SAM31_MEMORY_ATTENTION_DECODER_PY_DEVICE }}

# Convert the SAM3.1 memory-attention encoder plus mask decoder into the local ggml container.
sam31-convert-memory-attention-decoder-ggml checkpoint=SAM31_CONTRACT_SAM31 out=SAM31_MEMORY_ATTENTION_DECODER_MODEL:
    mkdir -p "$(dirname "{{ out }}")"
    uv run convert_sam3_to_ggml.py --model {{ checkpoint }} --output {{ out }} --ftype {{ SAM31_MEMORY_ATTENTION_DECODER_FTYPE }} --sam31 --sam31-memory-attention-decoder-only

# Run the C++ SAM3.1 memory-attention -> mask-decoder bridge and compare with Python.
sam31-memory-attention-decoder-parity out=SAM31_MEMORY_ATTENTION_DECODER_COMPARE_OUT: sam31-memory-attention-decoder-case (build-target "sam31_memory_attention_decoder_case")
    just sam31-convert-memory-attention-decoder-ggml
    mkdir -p {{ SAM31_MEMORY_ATTENTION_DECODER_CPP_OUT }} {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_memory_attention_decoder_case \
        --model {{ SAM31_MEMORY_ATTENTION_DECODER_MODEL }} \
        --case {{ SAM31_MEMORY_ATTENTION_DECODER_OUT }} \
        --out {{ SAM31_MEMORY_ATTENTION_DECODER_CPP_OUT }} \
        --threads 4 \
        --{{ SAM31_MEMORY_ATTENTION_DECODER_DEVICE }}
    uv run scripts/compare_raw_tensor_dirs.py \
        --expected {{ SAM31_MEMORY_ATTENTION_DECODER_OUT }}/expected_cpp_layout \
        --actual {{ SAM31_MEMORY_ATTENTION_DECODER_CPP_OUT }} \
        --tensor sam31_mem_attn_output \
        --tensor masks \
        --tensor iou_pred \
        --tensor mask_tokens_out \
        --tensor object_score_logits \
        --max-abs-tolerance {{ SAM31_MEMORY_ATTENTION_DECODER_MAX_ABS_TOLERANCE }} \
        --mean-abs-tolerance {{ SAM31_MEMORY_ATTENTION_DECODER_MEAN_ABS_TOLERANCE }} \
        --out {{ out }}/summary.json

# Inspect the official Python SAM3.1 tracking session state contract.
sam31-tracking-state-inspect out=SAM31_TRACKING_STATE_OUT:
    mkdir -p {{ out }}
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/inspect_sam31_tracking_state.py \
        --repo-root . \
        --sam3-repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --frames {{ SAM31_TRACKING_STATE_FRAMES }} \
        --prompt-mode {{ SAM31_TRACKING_STATE_PROMPT_MODE }} \
        --prompt "{{ SAM31_TRACKING_STATE_PROMPT }}" \
        --box-xywh {{ SAM31_TRACKING_STATE_BOX_XYWH }} \
        --box-label {{ SAM31_TRACKING_STATE_BOX_LABEL }} \
        --max-stream-frames {{ SAM31_TRACKING_STATE_STREAM_FRAMES }} \
        --out {{ out }}/summary.json

# Run a C++ SAM3.1 smoke that initializes tracking from a detection mask and propagates one frame.
sam31-tracking-mask-init-smoke out=SAM31_TRACKING_MASK_INIT_OUT: (build-target "sam31_tracking_mask_init_smoke")
    mkdir -p {{ out }}
    set -o pipefail; ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_tracking_mask_init_smoke \
        --model {{ SAM31_TRACKING_MASK_INIT_MODEL }} \
        --threads {{ SAM31_TRACKING_MASK_INIT_THREADS }} \
        --{{ SAM31_TRACKING_MASK_INIT_DEVICE }} \
        --width {{ SAM31_TRACKING_MASK_INIT_WIDTH }} \
        --height {{ SAM31_TRACKING_MASK_INIT_HEIGHT }} \
        --mask-case {{ SAM31_TRACKING_MASK_INIT_CASE }} \
        --frame1-offset {{ SAM31_TRACKING_MASK_INIT_FRAME1_OFFSET }} \
        --warmup-runs {{ SAM31_TRACKING_MASK_INIT_WARMUP_RUNS }} \
        --out {{ out }} \
        |& tee {{ out }}/summary.log

# Run official Python SAM3.1 on the same synthetic mask-init sequence and compare with C++ output.
sam31-mask-sequence-python out=SAM31_MASK_SEQUENCE_PY_OUT: sam31-tracking-mask-init-smoke
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/run_sam31_mask_sequence_python.py \
        --repo-root . \
        --sam3-repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --cpp-mask {{ SAM31_TRACKING_MASK_INIT_OUT }}/frame1_mask.png \
        --width {{ SAM31_TRACKING_MASK_INIT_WIDTH }} \
        --height {{ SAM31_TRACKING_MASK_INIT_HEIGHT }} \
        --mask-case {{ SAM31_TRACKING_MASK_INIT_CASE }} \
        --frame1-offset {{ SAM31_TRACKING_MASK_INIT_FRAME1_OFFSET }} \
        --dtype {{ SAM31_MASK_SEQUENCE_PY_DTYPE }} \
        --tf32 {{ SAM31_MASK_SEQUENCE_PY_TF32 }} \
        --warmup-runs {{ SAM31_MASK_SEQUENCE_PY_WARMUP_RUNS }}

# Run a SAM3.1 C++/Python mask-init kernel-variant matrix and summarize timing/parity.
sam31-mask-init-matrix out=SAM31_MASK_INIT_MATRIX_OUT: (build-target "sam31_tracking_mask_init_smoke")
    size_args=""; IFS=, read -ra sizes <<< "{{ SAM31_MASK_INIT_MATRIX_SIZES }}"; for size in "${sizes[@]}"; do size_args="${size_args} --size ${size}"; done; \
    case_args=""; IFS=, read -ra cases <<< "{{ SAM31_MASK_INIT_MATRIX_CASES }}"; for case in "${cases[@]}"; do case_args="${case_args} --case ${case}"; done; \
    variant_args=""; IFS=, read -ra variants <<< "{{ SAM31_MASK_INIT_MATRIX_VARIANTS }}"; for variant in "${variants[@]}"; do variant_args="${variant_args} --variant ${variant}"; done; \
    interleave_args=""; if [[ "{{ SAM31_MASK_INIT_MATRIX_INTERLEAVE }}" != "0" ]]; then interleave_args="--interleave-variants"; fi; \
    profile_args=""; if [[ "{{ SAM31_MASK_INIT_MATRIX_PROFILE_DEFAULT }}" != "0" ]]; then profile_args="--profile-default --profile-stage-top {{ SAM31_MASK_INIT_MATRIX_PROFILE_STAGE_TOP }} --profile-node-top {{ SAM31_MASK_INIT_MATRIX_PROFILE_NODE_TOP }}"; fi; \
    if [[ -n "{{ SAM31_MASK_INIT_MATRIX_PROFILE_WARMUP_RUNS }}" ]]; then profile_args="${profile_args} --profile-warmup-runs {{ SAM31_MASK_INIT_MATRIX_PROFILE_WARMUP_RUNS }}"; fi; \
    uv run --no-project scripts/run_sam31_mask_init_matrix.py \
        --repo-root . \
        --cpp-exe ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_tracking_mask_init_smoke \
        --model {{ SAM31_TRACKING_MASK_INIT_MODEL }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --sam3-repo {{ SAM3_REPO }} \
        --out {{ out }} \
        --repeats {{ SAM31_MASK_INIT_MATRIX_REPEATS }} \
        --threads {{ SAM31_TRACKING_MASK_INIT_THREADS }} \
        --device {{ SAM31_TRACKING_MASK_INIT_DEVICE }} \
        --python-dtype {{ SAM31_MASK_SEQUENCE_PY_DTYPE }} \
        --tf32 {{ SAM31_MASK_SEQUENCE_PY_TF32 }} \
        --python-warmup-runs {{ SAM31_MASK_INIT_MATRIX_PY_WARMUP_RUNS }} \
        --cpp-warmup-runs {{ SAM31_MASK_INIT_MATRIX_CPP_WARMUP_RUNS }} \
        ${interleave_args} ${profile_args} ${size_args} ${case_args} ${variant_args}

# Summarize SAM3.1 mask-init E2E split, Python comparison, and variant decisions.
sam31-mask-init-audit out=SAM31_MASK_INIT_AUDIT_OUT: sam31-mask-init-matrix
    mkdir -p {{ out }}
    uv run --no-project scripts/summarize_sam31_mask_init_matrix.py \
        --matrix {{ SAM31_MASK_INIT_MATRIX_OUT }}/summary.json \
        --out {{ out }}/summary.json \
        --markdown {{ out }}/optimization_targets.md

# Dump official Python SAM3.1 state immediately after the synthetic mask-init frame.
sam31-mask-init-state-python out=SAM31_MASK_INIT_STATE_PY_OUT:
    PYTHONPATH={{ SAM3_REPO }} uv run scripts/dump_sam31_mask_init_state.py \
        --repo-root . \
        --sam3-repo {{ SAM3_REPO }} \
        --checkpoint {{ SAM31_CONTRACT_SAM31 }} \
        --out {{ out }} \
        --width {{ SAM31_TRACKING_MASK_INIT_WIDTH }} \
        --height {{ SAM31_TRACKING_MASK_INIT_HEIGHT }} \
        --mask-case {{ SAM31_TRACKING_MASK_INIT_CASE }} \
        --frame1-offset {{ SAM31_TRACKING_MASK_INIT_FRAME1_OFFSET }} \
        --dtype {{ SAM31_MASK_SEQUENCE_PY_DTYPE }} \
        --tf32 {{ SAM31_MASK_SEQUENCE_PY_TF32 }}

# Dump C++ SAM3.1 state immediately after the synthetic mask-init frame.
sam31-mask-init-state-cpp out=SAM31_MASK_INIT_STATE_CPP_OUT: (build-target "sam31_tracking_mask_init_smoke")
    mkdir -p {{ out }}
    rm -f {{ out }}/input_image_preprocessed.bin {{ out }}/input_image_preprocessed.shape
    rm -f {{ out }}/vit_output.bin {{ out }}/vit_output.shape
    rm -f {{ out }}/vit_patch_embed.bin {{ out }}/vit_patch_embed.shape
    rm -f {{ out }}/vit_after_pos.bin {{ out }}/vit_after_pos.shape
    rm -f {{ out }}/vit_block_00_input.bin {{ out }}/vit_block_00_input.shape
    rm -f {{ out }}/vit_block_00_out.bin {{ out }}/vit_block_00_out.shape
    rm -f {{ out }}/vit_block_01_out.bin {{ out }}/vit_block_01_out.shape
    rm -f {{ out }}/vit_block_02_out.bin {{ out }}/vit_block_02_out.shape
    rm -f {{ out }}/vit_block_07_out.bin {{ out }}/vit_block_07_out.shape
    rm -f {{ out }}/vit_block_15_out.bin {{ out }}/vit_block_15_out.shape
    rm -f {{ out }}/vit_block_23_out.bin {{ out }}/vit_block_23_out.shape
    rm -f {{ out }}/vit_block_31_out.bin {{ out }}/vit_block_31_out.shape
    SAM3_DUMP_MEM_SLOT_DIR={{ out }} \
        SAM31_MASK_INIT_STATE_DUMP_DIR={{ out }} \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam31_tracking_mask_init_smoke \
        --model {{ SAM31_TRACKING_MASK_INIT_MODEL }} \
        --threads {{ SAM31_TRACKING_MASK_INIT_THREADS }} \
        --{{ SAM31_TRACKING_MASK_INIT_DEVICE }} \
        --width {{ SAM31_TRACKING_MASK_INIT_WIDTH }} \
        --height {{ SAM31_TRACKING_MASK_INIT_HEIGHT }} \
        --mask-case {{ SAM31_TRACKING_MASK_INIT_CASE }} \
        --frame1-offset {{ SAM31_TRACKING_MASK_INIT_FRAME1_OFFSET }} \
        --warmup-runs {{ SAM31_TRACKING_MASK_INIT_WARMUP_RUNS }} \
        --out {{ out }}/smoke

# Compare official Python and C++ SAM3.1 mask-init memory state dumps.
sam31-mask-init-state-compare out=SAM31_MASK_INIT_STATE_COMPARE_OUT: sam31-mask-init-state-python sam31-mask-init-state-cpp
    mkdir -p {{ out }}
    uv run scripts/compare_raw_tensor_dirs.py \
        --expected {{ SAM31_MASK_INIT_STATE_PY_OUT }}/expected_cpp_layout \
        --actual {{ SAM31_MASK_INIT_STATE_CPP_OUT }} \
        --tensor input_image_preprocessed \
        --tensor vit_patch_embed \
        --tensor vit_after_pos \
        --tensor vit_block_00_input \
        --tensor vit_block_00_out \
        --tensor vit_block_01_out \
        --tensor vit_block_02_out \
        --tensor vit_block_07_out \
        --tensor vit_block_15_out \
        --tensor vit_block_23_out \
        --tensor vit_block_31_out \
        --tensor vit_output \
        --tensor memory_mux_mask_interpolated \
        --tensor stored_spatial_feats \
        --tensor stored_spatial_pe \
        --tensor image_features \
        --tensor image_pos_enc \
        --max-abs-tolerance 5e-1 \
        --mean-abs-tolerance 5e-2 \
        --out {{ out }}/summary.json

# Summarize current active SAM3/SAM3.1 goal status from freshly generated local evidence.
sam3-sam31-goal-audit out=SAM3_SAM31_GOAL_AUDIT_OUT: cuda-health sam31-contract sam31-multiplex-parity sam31-mux-mask-decoder-parity sam31-memory-backbone-parity sam31-propagation-features-parity sam31-memory-attention-parity sam31-memory-attention-decoder-parity sam31-tracking-state-inspect sam31-mask-sequence-python sam31-mask-init-audit
    mkdir -p {{ out }}
    uv run scripts/summarize_sam3_sam31_goal_audit.py \
        --cuda-health {{ CUDA_HEALTH_OUT }}/summary.env \
        --sam31-coverage {{ SAM31_CONTRACT_OUT }}/sam3-vs-sam31-renamed-coverage.json \
        --sam31-multiplex {{ SAM31_MULTIPLEX_STATE_OUT }}/summary.json \
        --sam31-mux-mask-decoder-case {{ SAM31_MUX_MASK_DECODER_OUT }}/summary.json \
        --sam31-mux-mask-decoder-compare {{ SAM31_MUX_MASK_DECODER_COMPARE_OUT }}/summary.json \
        --sam31-memory-backbone-case {{ SAM31_MEMORY_BACKBONE_OUT }}/summary.json \
        --sam31-memory-backbone-compare {{ SAM31_MEMORY_BACKBONE_COMPARE_OUT }}/summary.json \
        --sam31-propagation-features-compare {{ SAM31_PROPAGATION_FEATURES_COMPARE_OUT }}/summary.json \
        --sam31-memory-attention-compare {{ SAM31_MEMORY_ATTENTION_COMPARE_OUT }}/summary.json \
        --sam31-memory-attention-decoder-compare {{ SAM31_MEMORY_ATTENTION_DECODER_COMPARE_OUT }}/summary.json \
        --sam31-tracking-state {{ SAM31_TRACKING_STATE_OUT }}/summary.json \
        --sam31-tracking-mask-init-smoke-log {{ SAM31_TRACKING_MASK_INIT_OUT }}/summary.log \
        --sam31-mask-sequence-python {{ SAM31_MASK_SEQUENCE_PY_OUT }}/summary.json \
        --sam31-mask-init-audit {{ SAM31_MASK_INIT_AUDIT_OUT }}/summary.json \
        --sam3-e2e-audit {{ SAM3_E2E_AUDIT }} \
        --sam3-perf-summary {{ SAM3_PERF_SUMMARY }} \
        {{ SAM3_PERF_SUMMARY_EXTRA_ARGS }} \
        --out {{ out }}/summary.json

# Run the q4_1 Hiera MLP fusion acceptance microbench.
fusion-kernel-acceptance out=FUSION_OUT: (build-target "sam3_mmq_bench")
    mkdir -p {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_mmq_bench \
        --cuda --type {{ FUSION_TYPE }} --k 448 --rows 1792 --fc2-rows 448 --cols 128 \
        --two-stage --bias --check --warmup 1 --iters 3 --tolerance {{ FUSION_TOLERANCE }} \
        > {{ out }}/q41_two_stage_no_cont_check.log
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_mmq_bench \
        --cuda --type {{ FUSION_TYPE }} --k 448 --rows 1792 --fc2-rows 448 --cols 4096 \
        --two-stage --bias --warmup 5 --iters 100 \
        > {{ out }}/q41_two_stage_no_cont_timing.log
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_mmq_bench \
        --cuda --type {{ FUSION_TYPE }} --k 448 --rows 1792 --fc2-rows 448 --cols 4096 \
        --two-stage --two-stage-cont --bias --warmup 5 --iters 100 \
        > {{ out }}/q41_two_stage_cont_timing.log

# Run paired q4_1 two-stage fusion baselines under the same build and warm device state.
fusion-kernel-paired out=FUSION_OUT repeats=FUSION_REPEATS: (build-target "sam3_mmq_bench")
    mkdir -p {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_mmq_bench \
        --cuda --type {{ FUSION_TYPE }} --k 448 --rows 1792 --fc2-rows 448 --cols 4096 \
        --two-stage --bias --warmup 10 --iters 20 \
        > {{ out }}/warmup.log
    for i in $(seq 1 {{ repeats }}); do \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_mmq_bench \
            --cuda --type {{ FUSION_TYPE }} --k 448 --rows 1792 --fc2-rows 448 --cols 4096 \
            --two-stage --bias --warmup 5 --iters 100 \
            > {{ out }}/default_$i.log; \
        GGML_CUDA_ENABLE_MMQ_PREQUANT_CACHE=1 \
        ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_mmq_bench \
            --cuda --type {{ FUSION_TYPE }} --k 448 --rows 1792 --fc2-rows 448 --cols 4096 \
            --two-stage --bias --warmup 5 --iters 100 \
            > {{ out }}/prequant_$i.log; \
    done
    uv run scripts/summarize_fusion_paired.py {{ out }} --drop-first --out {{ out }}/summary_drop_first.json

# Run the local bias+GELU+q8_1 DS4 fusion kernel microbench for the Hiera MLP fc1 shape.
gelu-quant-fusion-bench out=FUSION_OUT: (build-target "sam3_f32_batched_bench")
    mkdir -p {{ out }}
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_f32_batched_bench \
        --cuda --gelu-quant-ds4 --bias --k 1792 --cols 4096 --batches 1 \
        --warmup 10 --iters 50 \
        > {{ out }}/bias_gelu_quant_ds4.log
    ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_f32_batched_bench \
        --cuda --gelu-quant-ds4 --no-bias --k 1792 --cols 4096 --batches 1 \
        --warmup 10 --iters 50 \
        > {{ out }}/gelu_quant_ds4.log
