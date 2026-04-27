#!/bin/bash
cmake_minimum_required(VERSION 2.8.12)
project(fastbev)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})


find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV)

set(DEFAULT_STB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/stb")
set(DEFAULT_PYBIND11_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/pybind11")
set(DEFAULT_CUOSD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/cuOSD")
set(DEFAULT_SPCONV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/3DSparseConvolution/libspconv")

if(DEFINED ENV{STB_ROOT} AND EXISTS "$ENV{STB_ROOT}")
  set(STB_ROOT "$ENV{STB_ROOT}")
elseif(EXISTS "${DEFAULT_STB_ROOT}")
  set(STB_ROOT "${DEFAULT_STB_ROOT}")
elseif(EXISTS "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/stb")
  set(STB_ROOT "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/stb")
endif()

if(DEFINED ENV{PYBIND11_ROOT} AND EXISTS "$ENV{PYBIND11_ROOT}")
  set(PYBIND11_ROOT "$ENV{PYBIND11_ROOT}")
elseif(EXISTS "${DEFAULT_PYBIND11_ROOT}")
  set(PYBIND11_ROOT "${DEFAULT_PYBIND11_ROOT}")
endif()

if(DEFINED ENV{CUOSD_ROOT} AND EXISTS "$ENV{CUOSD_ROOT}")
  set(CUOSD_ROOT "$ENV{CUOSD_ROOT}")
elseif(EXISTS "${DEFAULT_CUOSD_ROOT}")
  set(CUOSD_ROOT "${DEFAULT_CUOSD_ROOT}")
elseif(EXISTS "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/cuOSD")
  set(CUOSD_ROOT "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/cuOSD")
endif()

if(DEFINED ENV{SPCONV_ROOT} AND EXISTS "$ENV{SPCONV_ROOT}")
  set(SPCONV_ROOT "$ENV{SPCONV_ROOT}")
elseif(EXISTS "${DEFAULT_SPCONV_ROOT}")
  set(SPCONV_ROOT "${DEFAULT_SPCONV_ROOT}")
elseif(EXISTS "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/3DSparseConvolution/libspconv")
  set(SPCONV_ROOT "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/3DSparseConvolution/libspconv")
endif()

if(NOT DEFINED SPCONV_ROOT OR NOT EXISTS "${SPCONV_ROOT}/lib/${arch}/libspconv.so")
  message(FATAL_ERROR "libspconv.so not found. Set SPCONV_ROOT to a libspconv directory that contains include/ and lib/${arch}/libspconv.so")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  # ${OpenCV_INCLUDE_DIRS}
  ${STB_ROOT}
  ${PYBIND11_ROOT}/include
  ${CUOSD_ROOT}/src
  ${SPCONV_ROOT}/include
  src/common
  src/
  src/tracking
)

link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ${SPCONV_ROOT}/lib/${arch}
  build
  $ENV{Python_Lib}
)

file(GLOB_RECURSE CORE_FILES 
  src/fastbev/*.cu 
  src/fastbev/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
  src/common/stb_image_impl.cpp
  src/camera/camera_frame.cpp
  src/pipeline/perception_pipeline.cpp
  src/tracking/tracker.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ${CUOSD_ROOT}/src/*.cpp
  ${CUOSD_ROOT}/src/*.cu
  src/common/visualize.cu
)

####################### libfastbev_core.so ##########################
cuda_add_library(fastbev_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(fastbev_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)
#######################################################################


####################### fastbev program #############################
cuda_add_executable(fastbev 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(fastbev
  fastbev_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)
#######################################################################

########################## libpybev.so ################################
if(${CMAKE_BUILD_PYTHON} STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )

  target_link_libraries(pybev
    fastbev_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
    ${OpenCV_LIBS}
  )
endif()
#######################################################################
