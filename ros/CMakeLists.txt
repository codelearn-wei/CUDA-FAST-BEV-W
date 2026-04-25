cmake_minimum_required(VERSION 3.10)
project(fastbev_ros)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# ─── ROS 依赖 ──────────────────────────────────────────────────────────────
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  visualization_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
)

# ─── CUDA ─────────────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)

# ─── OpenCV ───────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ─── catkin 声明 ───────────────────────────────────────────────────────────
catkin_package(
  CATKIN_DEPENDS
    roscpp sensor_msgs std_msgs visualization_msgs
    geometry_msgs cv_bridge image_transport message_filters
)

# ─── FastBEV 核心库路径 ────────────────────────────────────────────────────
# 假设本 ros/ 目录与 src/ 同级
set(FASTBEV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(FASTBEV_BUILD "${FASTBEV_ROOT}/build")

# 依赖库路径（与顶层 CMakeLists.txt 保持一致）
set(DEFAULT_CUOSD_ROOT   "${FASTBEV_ROOT}/../libraries/cuOSD")
set(FALLBACK_CUOSD_ROOT  "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/cuOSD")
if(EXISTS "${DEFAULT_CUOSD_ROOT}")
  set(CUOSD_ROOT "${DEFAULT_CUOSD_ROOT}")
elseif(EXISTS "${FALLBACK_CUOSD_ROOT}")
  set(CUOSD_ROOT "${FALLBACK_CUOSD_ROOT}")
else()
  message(FATAL_ERROR "cuOSD 未找到，请设置 CUOSD_ROOT")
endif()

set(DEFAULT_SPCONV_ROOT  "${FASTBEV_ROOT}/../libraries/3DSparseConvolution/libspconv")
set(FALLBACK_SPCONV_ROOT "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/3DSparseConvolution/libspconv")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(EXISTS "${DEFAULT_SPCONV_ROOT}/lib/${arch}/libspconv.so")
  set(SPCONV_ROOT "${DEFAULT_SPCONV_ROOT}")
elseif(EXISTS "${FALLBACK_SPCONV_ROOT}/lib/${arch}/libspconv.so")
  set(SPCONV_ROOT "${FALLBACK_SPCONV_ROOT}")
else()
  message(FATAL_ERROR "libspconv.so 未找到，请设置 SPCONV_ROOT")
endif()

# ─── include 目录 ─────────────────────────────────────────────────────────
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${FASTBEV_ROOT}/src          # fastbev/*.hpp + common/*.hpp
  ${CUOSD_ROOT}/src
  ${SPCONV_ROOT}/include
  $ENV{TensorRT_Inc}
)

# ─── 链接目录 ─────────────────────────────────────────────────────────────
link_directories(
  ${FASTBEV_BUILD}             # libfastbev_core.so
  ${SPCONV_ROOT}/lib/${arch}
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
)

# ─── 编译节点 ─────────────────────────────────────────────────────────────
cuda_add_executable(fastbev_ros_node
  fastbev_ros_node.cpp
)

target_compile_options(fastbev_ros_node PRIVATE
  -std=c++14 -O3 -Wno-deprecated-declarations
)

target_link_libraries(fastbev_ros_node
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  fastbev_core         # 来自 ${FASTBEV_BUILD}/libfastbev_core.so
  nvinfer
  nvinfer_plugin
  nvonnxparser
  spconv
  cudart
)

# ─── 安装 ─────────────────────────────────────────────────────────────────
install(TARGETS fastbev_ros_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
