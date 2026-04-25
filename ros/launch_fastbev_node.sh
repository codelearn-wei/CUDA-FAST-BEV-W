#!/bin/bash
# CUDA-FastBEV ROS 节点启动脚本
# 用法: source /opt/ros/noetic/setup.bash && bash ros/launch_fastbev_node.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 环境变量（与顶层 tool/environment.sh 保持一致）──────────────────────
source "$PROJECT_ROOT/tool/environment.sh" 2>/dev/null || true
export LD_LIBRARY_PATH="$PROJECT_ROOT/build:$LD_LIBRARY_PATH"

# ── 默认参数 ────────────────────────────────────────────────────────────
MODEL="${FASTBEV_MODEL:-resnet18}"
SCORE_THR="${FASTBEV_SCORE_THR:-0.5}"
GEOMETRY_DIR="${FASTBEV_GEOMETRY_DIR:-$PROJECT_ROOT/example-data/example-data}"
CLASS_FILTER="${FASTBEV_CLASS_FILTER:-}"  # 空=全部

echo "========================================="
echo "  CUDA-FastBEV ROS 节点"
echo "  模型:      $MODEL"
echo "  阈值:      $SCORE_THR"
echo "  几何目录:  $GEOMETRY_DIR"
echo "========================================="

# 运行节点
rosrun fastbev_ros fastbev_ros_node \
  _model:="$MODEL" \
  _score_thr:="$SCORE_THR" \
  _geometry_dir:="$GEOMETRY_DIR" \
  _class_filter:="$CLASS_FILTER" \
  "$@"
