cmake_minimum_required(VERSION 3.9)
project(cuda_orb)

find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(cuda_orb main.cpp src/gpu_feature_processor.cpp src/feature_processors.cpp)
include_directories(include)

target_link_libraries(cuda_orb ${OpenCV_LIBS} pthread)

set_property(TARGET cuda_orb PROPERTY CUDA_ARCHITECTURES 61)
