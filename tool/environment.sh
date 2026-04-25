#!/bin/bash

# Optional manual overrides:
# export TensorRT_ROOT=/path/to/TensorRT
# export TensorRT_Lib=/path/to/TensorRT/lib
# export TensorRT_Inc=/path/to/TensorRT/include
# export TensorRT_Bin=/path/to/TensorRT/bin

append_path() {
    if [ -n "$1" ] && [ -d "$1" ]; then
        case ":$PATH:" in
            *":$1:"*) ;;
            *) export PATH="$1:$PATH" ;;
        esac
    fi
}

append_ld_library_path() {
    if [ -n "$1" ] && [ -d "$1" ]; then
        case ":$LD_LIBRARY_PATH:" in
            *":$1:"*) ;;
            *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
}

append_python_path() {
    if [ -n "$1" ] && [ -d "$1" ]; then
        case ":$PYTHONPATH:" in
            *":$1:"*) ;;
            *) export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
    fi
}

if [ -n "$TensorRT_Bin" ] && [ ! -x "${TensorRT_Bin}/trtexec" ]; then
    unset TensorRT_Bin TensorRT_Lib TensorRT_Inc TensorRT_ROOT
fi

if [ -z "$TensorRT_ROOT" ]; then
    for candidate in \
        "/home/dfg-autoware/TensorRT-8.5.2.2" \
        "/usr/src/tensorrt" \
        "/opt/tensorrt" \
        "/usr/local/TensorRT"; do
        if [ -x "${candidate}/bin/trtexec" ]; then
            export TensorRT_ROOT="$candidate"
            break
        fi
    done
fi

if [ -n "$TensorRT_ROOT" ]; then
    export TensorRT_Bin="${TensorRT_Bin:-$TensorRT_ROOT/bin}"
    export TensorRT_Lib="${TensorRT_Lib:-$TensorRT_ROOT/lib}"
    export TensorRT_Inc="${TensorRT_Inc:-$TensorRT_ROOT/include}"
fi

if [ -z "$SPCONV_ROOT" ]; then
    for candidate in \
        "$(pwd)/../libraries/3DSparseConvolution/libspconv" \
        "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/3DSparseConvolution/libspconv"; do
        if [ -f "${candidate}/lib/x86_64/libspconv.so" ]; then
            export SPCONV_ROOT="$candidate"
            break
        fi
    done
fi

if [ -z "$CUOSD_ROOT" ]; then
    for candidate in \
        "$(pwd)/../libraries/cuOSD" \
        "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/cuOSD"; do
        if [ -d "${candidate}/src" ]; then
            export CUOSD_ROOT="$candidate"
            break
        fi
    done
fi

if [ -z "$STB_ROOT" ]; then
    for candidate in \
        "$(pwd)/../dependencies/stb" \
        "/home/dfg-autoware/perception/bevfusion_ws/src/BEVFusion-ROS-TensorRT/third_party/stb"; do
        if [ -f "${candidate}/stb_image.h" ]; then
            export STB_ROOT="$candidate"
            break
        fi
    done
fi

export CUDA_Lib=/usr/local/cuda/lib64
export CUDA_Inc=/usr/local/cuda/include
export CUDA_Bin=/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda/

export CUDNN_Lib=/usr/local/cuda/lib64


# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet18int8

# fp16/int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=example-data
export USE_Python=OFF

# check the configuration path
# clean the configuration status
export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "||  CUDNN: $CUDNN_Lib"
echo "||  SPCONV: $SPCONV_ROOT"
echo "||  CUOSD: $CUOSD_ROOT"
echo "||  STB: $STB_ROOT"
echo "=========================================================="

BuildDirectory=`pwd`/build

if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`
    echo Find Python_Inc: $Python_Inc
    echo Find Python_Lib: $Python_Lib
    echo Find Python_Soname: $Python_Soname
fi

append_path "$TensorRT_Bin"
append_path "$CUDA_Bin"
append_ld_library_path "$TensorRT_Lib"
append_ld_library_path "$CUDA_Lib"
append_ld_library_path "$CUDNN_Lib"
append_ld_library_path "$BuildDirectory"
append_python_path "$BuildDirectory"
export ConfigurationStatus=Success

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    echo "Current CUDA SM: $cudasm"
fi

export CUDASM=$cudasm

echo Configuration done!
