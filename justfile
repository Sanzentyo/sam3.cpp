set shell := ["bash", "-cu"]
set dotenv-load

BUILD_MODE := env_var_or_default("BUILD_MODE", "release")
CUDA_ARCH := env_var_or_default("CUDA_ARCH", "120")
CXX_COMPILER := env_var_or_default("CXX_COMPILER", "/usr/bin/g++-14")
CUDA_HOST_COMPILER := env_var_or_default("CUDA_HOST_COMPILER", "/usr/bin/g++-14")
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

CLANG_TIDY := env_var_or_default("CLANG_TIDY", "uvx --from clang-tidy clang-tidy")
CLANG_FORMAT := env_var_or_default("CLANG_FORMAT", "uvx --from clang-format clang-format")
CPPCHECK := env_var_or_default("CPPCHECK", "uvx --from cppcheck cppcheck")

# Default: list all recipes.
default:
    @just --list

# Configure xmake for the CUDA benchmark build.
config mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} -y

# Build sam3_benchmark via xmake.
build mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} -y
    xmake build sam3_benchmark

# Build a specific xmake target.
build-target target mode=BUILD_MODE:
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} -y
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
    xmake f -m {{ mode }} --cuda=y --cuda_arch={{ CUDA_ARCH }} --cuda_host_compiler={{ CUDA_HOST_COMPILER }} --cxx_compiler={{ CXX_COMPILER }} -y
    xmake build sam3_benchmark
    ln -sf build/xmake-{{ mode }}-cuda/compile_commands.json compile_commands.json

# Format files changed in the current worktree/index.
fmt:
    files="$({ git diff --name-only --diff-filter=ACMR; git diff --cached --name-only --diff-filter=ACMR; } | sort -u | grep -E '^(sam3\.(cpp|h)|examples/.*\.(cpp|h))$' || true)"; \
    if [ -n "$files" ]; then printf '%s\n' "$files" | xargs -r {{ CLANG_FORMAT }} -i; fi

# Format all project-owned C++ sources.
fmt-all:
    find sam3.cpp sam3.h examples -type f \( -name '*.cpp' -o -name '*.h' \) | xargs -r {{ CLANG_FORMAT }} -i

# Check formatting without modifying files.
fmt-check:
    find sam3.cpp sam3.h examples -type f \( -name '*.cpp' -o -name '*.h' \) | xargs -r {{ CLANG_FORMAT }} --dry-run -Werror

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
        sam3.cpp examples/benchmark.cpp examples/profile_edgetam.cpp examples/quantize.cpp examples/smoke_sam3.cpp

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
