#!/bin/bash
# =============================================================================
# setup_maptr_env.sh — 创建 MapTRv2 推理所需的 conda 环境 (maptr)
#
# 用法：
#   bash tools/maptr/setup_maptr_env.sh
#
# 该脚本在项目根目录下执行（CUDA-FastBEV/），会：
#   1. 创建 maptr conda 环境 (Python 3.8)
#   2. 安装 PyTorch 1.10.0 + CUDA 11.1
#   3. 安装 mmcv-full 1.4.0（兼容 MapTRv2）
#   4. 安装 mmdet 2.14.0 / mmsegmentation 0.14.1 / timm
#   5. 安装 MapTRv2 自带的 mmdetection3d（开发模式）
#   6. pip安装 onnx=1.10.0（可选，推理时需要）,protobuf=3.20.0（解决onnx安装后protobuf版本过高问题）
#   7. 编译 geometric_kernel_attn 自定义 CUDA 算子
#   8. 安装其他依赖（nuscenes-devkit, shapely, etc.）
#   9. 验证安装是否成功
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAPTR_REPO="${PROJECT_ROOT}/tools/MapTRv2"
OPS_DIR="${MAPTR_REPO}/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn"
ENV_NAME="maptr"

echo "========================================"
echo "  MapTRv2 环境安装脚本"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  MapTRv2 目录: ${MAPTR_REPO}"
echo "========================================"

# ── 1. 检查 conda 是否可用 ─────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[ERROR] 未找到 conda，请先安装 Anaconda/Miniconda"
    exit 1
fi

# ── 2. 如果环境已存在，询问是否重建 ──────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "[INFO] conda 环境 '${ENV_NAME}' 已存在。"
    read -p "是否删除并重新创建？[y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] 删除旧环境 ${ENV_NAME}..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "[INFO] 跳过创建，将在现有环境中继续安装..."
    fi
fi

# ── 3. 创建/激活环境 ─────────────────────────────────────────────────────────
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "[STEP 1/6] 创建 conda 环境 ${ENV_NAME} (Python 3.8)..."
    conda create -n "${ENV_NAME}" python=3.8 -y
fi

# ── 4. 安装 PyTorch ──────────────────────────────────────────────────────────
echo ""
echo "[STEP 2/6] 安装 PyTorch 1.10.0 + CUDA 11.1..."
conda run -n "${ENV_NAME}" pip install \
    torch==1.10.0+cu111 \
    torchvision==0.11.0+cu111 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

# ── 5. 安装 mmcv-full 1.4.0 ─────────────────────────────────────────────────
echo ""
echo "[STEP 3/6] 安装 mmcv-full 1.4.0 (兼容 mmcv 1.x API)..."
conda run -n "${ENV_NAME}" pip install \
    mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

# ── 6. 安装 onnx 1.10.0 和 protobuf 3.20.0（可选，推理时需要）────────────────────────────
echo ""
echo "[STEP 4/6] 安装 onnx 1.10.0 / protobuf 3.20.0..."
conda run -n "${ENV_NAME}" pip install \
    onnx==1.10.0 \
    protobuf==3.20.0

# ── 7. 安装 mmdet / mmseg / timm ────────────────────────────────────────────
echo ""
echo "[STEP 5/6] 安装 mmdet 2.14.0 / mmsegmentation 0.14.1 / timm..."
conda run -n "${ENV_NAME}" pip install \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    timm \
    einops \
    shapely==1.8.5 \
    nuscenes-devkit \
    numpy \
    opencv-python \
    Pillow \
    scipy \
    tqdm

# ── 7. 安装 MapTRv2 内置 mmdetection3d（开发模式）────────────────────────────
echo ""
echo "[STEP 6/6] 安装 MapTRv2 bundled mmdetection3d..."
cd "${MAPTR_REPO}/mmdetection3d"
conda run -n "${ENV_NAME}" pip install -v -e . \
    --no-build-isolation 2>&1 | tail -5
cd "${PROJECT_ROOT}"

# ── 8. 编译 geometric_kernel_attn CUDA 算子 ──────────────────────────────────
echo ""
echo "[STEP 7/6] 编译 geometric_kernel_attn CUDA 算子..."
cd "${OPS_DIR}"
conda run -n "${ENV_NAME}" python setup.py build_ext --inplace 2>&1 | tail -10
# 验证编译结果
if ls *.so 2>/dev/null | head -1 | grep -q ".so"; then
    echo "[OK] CUDA 算子编译成功: $(ls *.so)"
else
    echo "[WARNING] 未找到 .so 文件，尝试 install 模式..."
    conda run -n "${ENV_NAME}" python setup.py build install 2>&1 | tail -5
fi
cd "${PROJECT_ROOT}"

# ── 9. 验证安装 ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  验证安装..."
echo "========================================"
conda run -n "${ENV_NAME}" python -c "
import torch, mmcv, mmdet, mmseg
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('mmcv:', mmcv.__version__)
print('mmdet:', mmdet.__version__)
print('mmseg:', mmseg.__version__)

# 验证 mmcv.Config (mmcv 1.x API)
cfg = mmcv.Config({'a': 1})
assert cfg.a == 1
print('mmcv.Config: OK')

# 验证 mmdet3d
import mmdet3d
print('mmdet3d:', mmdet3d.__version__)

# 验证 MapTRv2 plugin
import sys
sys.path.insert(0, '${MAPTR_REPO}')
from projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn import (
    GeometricKernelAttentionFunc
)
print('GeometricKernelAttentionFunc: OK')
print('')
print('[SUCCESS] maptr 环境安装完成！')
" 2>&1

echo ""
echo "========================================"
echo "  安装完成！"
echo "  运行模型推理："
echo "    conda run -n bev python tools/maptr/run_maptr.py \\"
echo "      --frames-dir outputs/frames --mode model \\"
echo "      --ckpt model/maptr/maptr_nano_r18_110e.pth"
echo "========================================"
