cmake_minimum_required(VERSION 3.1)
project(ITS_test CUDA CXX)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES 80)

add_subdirectory(lib)

include_directories( 
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ 
    )

    add_executable(tile_comp
    tile_size/tile_size_test.cu 
    tile_size/cublas.cu
)

target_link_libraries(tile_comp
    culib
    cublas
)

target_compile_definitions(tile_comp PRIVATE TIME=1)

add_executable(tile_comp_prof
    tile_size/tile_size_test.cu 
    tile_size/cublas.cu
)

target_link_libraries(tile_comp_prof
    culib
    cublas
)