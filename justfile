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
MODEL_MATRIX_PYTHON_FAMILY := env_var_or_default("MODEL_MATRIX_PYTHON_FAMILY", "")
MODEL_MATRIX_PYTHON_SAM3_VERSION := env_var_or_default("MODEL_MATRIX_PYTHON_SAM3_VERSION", "none")
MODEL_MATRIX_PYTHON_DTYPE := env_var_or_default("MODEL_MATRIX_PYTHON_DTYPE", "bf16")
MODEL_MATRIX_TF32_POLICY := env_var_or_default("MODEL_MATRIX_TF32_POLICY", "on")
MODEL_MATRIX_SKIP_PYTHON_SAM2 := env_var_or_default("MODEL_MATRIX_SKIP_PYTHON_SAM2", "")
SAM3_REPO := env_var_or_default("SAM3_REPO", "external/sam3")
CUDNN_SDPA_OUT := env_var_or_default("CUDNN_SDPA_OUT", "outputs/cudnn-sdpa-head64")
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
SAM3_SAM31_GOAL_AUDIT_OUT := env_var_or_default("SAM3_SAM31_GOAL_AUDIT_OUT", "outputs/sam3-sam31-goal-audit")

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
        sam3.cpp sam3_multiplex.h sam3_sam31.h sam3_sam31_tensor_manifest.h examples/benchmark.cpp examples/profile_edgetam.cpp examples/quantize.cpp examples/smoke_sam3.cpp examples/sam31_multiplex_state.cpp examples/sam31_mux_mask_decoder_case.cpp

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
        --python-dtype {{ MODEL_MATRIX_PYTHON_DTYPE }} \
        --tf32-policy {{ MODEL_MATRIX_TF32_POLICY }} \
        --python-sam3-version {{ MODEL_MATRIX_PYTHON_SAM3_VERSION }} \
        ${MODEL_MATRIX_SKIP_PYTHON_SAM2:+--skip-python-sam2} \
        ${MODEL_MATRIX_PYTHON_FAMILY:+--python-family "$MODEL_MATRIX_PYTHON_FAMILY"} \
        ${MODEL_MATRIX_FILTER:+--filter "$MODEL_MATRIX_FILTER"}

# Measure cuDNN SDPA against the two SAM3 ViT head64 attention shapes and a fresh ggml FATTN profile.
cudnn-sdpa-head64 out=CUDNN_SDPA_OUT: build (build-target "sam3_cudnn_sdpa_bench")
    mkdir -p {{ out }}
    GGML_CUDA_PROFILE_FATTN=1 ./build/xmake-{{ BUILD_MODE }}-cuda/examples/sam3_benchmark \
        --models-dir {{ MODELS_DIR }} \
        --video {{ VIDEO }} \
        --gpu-only --bbox-only --quiet --n-frames 3 \
        --point-x 315 --point-y 250 --text-prompt person \
        --filter sam3-bf16 --no-isolation \
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

# Summarize current active SAM3/SAM3.1 goal status from freshly generated local evidence.
sam3-sam31-goal-audit out=SAM3_SAM31_GOAL_AUDIT_OUT: cuda-health sam31-contract sam31-multiplex-parity sam31-mux-mask-decoder-case
    mkdir -p {{ out }}
    uv run scripts/summarize_sam3_sam31_goal_audit.py \
        --cuda-health {{ CUDA_HEALTH_OUT }}/summary.env \
        --sam31-coverage {{ SAM31_CONTRACT_OUT }}/sam3-vs-sam31-renamed-coverage.json \
        --sam31-multiplex {{ SAM31_MULTIPLEX_STATE_OUT }}/summary.json \
        --sam31-mux-mask-decoder-case {{ SAM31_MUX_MASK_DECODER_OUT }}/summary.json \
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
