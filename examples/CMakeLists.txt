# Example executables for sam3.cpp

# Quantization tool (no GUI dependency)
add_executable(sam3_quantize quantize.cpp)
target_link_libraries(sam3_quantize PRIVATE ggml)

add_executable(sam3_benchmark benchmark.cpp)
target_link_libraries(sam3_benchmark PRIVATE sam3)

add_executable(sam3_smoke smoke_sam3.cpp)
target_link_libraries(sam3_smoke PRIVATE sam3)

add_executable(sam3_profile_edgetam profile_edgetam.cpp)
target_link_libraries(sam3_profile_edgetam PRIVATE sam3)

add_executable(sam3_fattn_parity fattn_parity.cpp)
target_link_libraries(sam3_fattn_parity PRIVATE sam3)

add_executable(sam3_mmq_bench mmq_bench.cpp)
target_link_libraries(sam3_mmq_bench PRIVATE sam3)

add_executable(sam31_multiplex_state sam31_multiplex_state.cpp)
target_include_directories(sam31_multiplex_state PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(sam31_multiplex_state PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_mux_mask_decoder_case sam31_mux_mask_decoder_case.cpp)
target_link_libraries(sam31_mux_mask_decoder_case PRIVATE sam3)
set_target_properties(sam31_mux_mask_decoder_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_memory_backbone_case sam31_memory_backbone_case.cpp)
target_link_libraries(sam31_memory_backbone_case PRIVATE sam3)
set_target_properties(sam31_memory_backbone_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_propagation_features_case sam31_propagation_features_case.cpp)
target_link_libraries(sam31_propagation_features_case PRIVATE sam3)
set_target_properties(sam31_propagation_features_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_memory_attention_case sam31_memory_attention_case.cpp)
target_link_libraries(sam31_memory_attention_case PRIVATE sam3)
set_target_properties(sam31_memory_attention_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_memory_attention_decoder_case sam31_memory_attention_decoder_case.cpp)
target_link_libraries(sam31_memory_attention_decoder_case PRIVATE sam3)
set_target_properties(sam31_memory_attention_decoder_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_tracking_mask_init_smoke sam31_tracking_mask_init_smoke.cpp)
target_link_libraries(sam31_tracking_mask_init_smoke PRIVATE sam3)
set_target_properties(sam31_tracking_mask_init_smoke PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_vit_prefix_case sam31_vit_prefix_case.cpp)
target_link_libraries(sam31_vit_prefix_case PRIVATE sam3)
set_target_properties(sam31_vit_prefix_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam31_vit_block_case sam31_vit_block_case.cpp)
target_link_libraries(sam31_vit_block_case PRIVATE sam3)
set_target_properties(sam31_vit_block_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam3_vit_stage_capture sam3_vit_stage_capture.cpp)
target_link_libraries(sam3_vit_stage_capture PRIVATE sam3)
set_target_properties(sam3_vit_stage_capture PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam3_text_encoder_case sam3_text_encoder_case.cpp)
target_link_libraries(sam3_text_encoder_case PRIVATE sam3)
set_target_properties(sam3_text_encoder_case PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(sam3_vit_batch_bench sam3_vit_batch_bench.cpp)
target_link_libraries(sam3_vit_batch_bench PRIVATE sam3)
set_target_properties(sam3_vit_batch_bench PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(GGML_CUDA)
    add_executable(sam3_mmq_q8_layout_parity mmq_q8_layout_parity.cu)
    set_source_files_properties(mmq_q8_layout_parity.cu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    set_target_properties(sam3_mmq_q8_layout_parity PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    target_include_directories(sam3_mmq_q8_layout_parity PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/ggml/src
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda
    )
    target_link_libraries(sam3_mmq_q8_layout_parity PRIVATE ggml-cuda ggml)
endif()

if(GGML_CUDA)
    add_executable(sam3_f32_batched_bench f32_batched_bench.cpp f32_batched_custom.cu)
    set_source_files_properties(f32_batched_custom.cu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    target_link_libraries(sam3_f32_batched_bench PRIVATE cublasLt)
else()
    add_executable(sam3_f32_batched_bench f32_batched_bench.cpp)
endif()
target_link_libraries(sam3_f32_batched_bench PRIVATE sam3)
set_target_properties(sam3_f32_batched_bench PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)

if(GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    find_path(CUDNN_FRONTEND_INCLUDE_DIR
        NAMES cudnn_frontend.h
        PATHS
            ${CMAKE_SOURCE_DIR}/external/cudnn-frontend/include
            ${CMAKE_SOURCE_DIR}/../cudnn-frontend/include
    )
    find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn.h
        PATHS /usr/include/x86_64-linux-gnu /usr/local/cuda/include
    )
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
    )

    if(CUDAToolkit_FOUND AND CUDNN_FRONTEND_INCLUDE_DIR AND CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        add_executable(sam3_cudnn_sdpa_bench cudnn_sdpa_bench.cpp cudnn_sdpa_convert.cu)
        target_include_directories(sam3_cudnn_sdpa_bench PRIVATE
            ${CUDNN_FRONTEND_INCLUDE_DIR}
            ${CUDNN_INCLUDE_DIR}
        )
        target_link_libraries(sam3_cudnn_sdpa_bench PRIVATE
            ${CUDNN_LIBRARY}
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
        )
        set_target_properties(sam3_cudnn_sdpa_bench PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
        )

        add_executable(sam3_cudnn_mlp_bench cudnn_mlp_bench.cpp)
        target_include_directories(sam3_cudnn_mlp_bench PRIVATE
            ${CUDNN_FRONTEND_INCLUDE_DIR}
            ${CUDNN_INCLUDE_DIR}
        )
        target_link_libraries(sam3_cudnn_mlp_bench PRIVATE
            ${CUDNN_LIBRARY}
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
        )
        set_target_properties(sam3_cudnn_mlp_bench PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
    else()
        message(STATUS "cuDNN Frontend or cuDNN not found -- skipping cuDNN benches")
    endif()
endif()

# SDL2 + ImGui are optional -- build only if SDL2 is found.
find_package(SDL2 QUIET)

if(SDL2_FOUND)
    add_subdirectory(third-party)

    add_executable(sam3_image main_image.cpp)
    target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

    add_executable(sam3_video main_video.cpp)
    target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
else()
    message(STATUS "SDL2 not found -- skipping GUI examples")
endif()
