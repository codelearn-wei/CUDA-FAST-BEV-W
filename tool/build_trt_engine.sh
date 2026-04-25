#!/bin/bash
# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# tensorrt version
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

function get_onnx_number_io(){

    # $1=model
    model=$1

    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    number_of_input=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    need_output_flg=$5
    result_save_directory=$base/build
    onnx=$base/$name.onnx

    if [ -f "${result_save_directory}/$name.plan" ]; then
        echo Model ${result_save_directory}/$name.plan already build 🙋🙋🙋.
        return
    fi
    
    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    echo $number_of_input  $number_of_output

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    if [ "$need_output_flg" == "need" ]; then
        cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} \
            --saveEngine=${result_save_directory}/$name.plan \
            --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
            --dumpProfile --separateProfileRun \
            --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"
    else
        cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} \
            --saveEngine=${result_save_directory}/$name.plan \
            --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
            --dumpProfile --separateProfileRun \
            --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"
    fi
    echo $cmd
    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd > ${result_save_directory}/$name.log 2>&1
}

function precision_flags_for_model() {
    model_name=$1
    precision=$2

    if [ "$precision" == "int8" ]; then
        echo "--fp16 --int8"
    else
        echo "--fp16"
    fi
}

function default_precision_for_model() {
    model_name=$1
    if [[ "$model_name" == *int8* ]]; then
        echo "int8"
    else
        echo "fp16"
    fi
}

function build_one_model_family() {
    model_name=$1
    precision=$2

    base=model/$model_name
    if [ ! -f "$base/fastbev_pre_trt.onnx" ] || [ ! -f "$base/fastbev_post_trt_decode.onnx" ]; then
        echo "Skip $model_name because required onnx files are missing."
        return
    fi

    trtexec_dynamic_flags=$(precision_flags_for_model "$model_name" "$precision")

    echo "=========================================================="
    echo "Build model family: $model_name"
    echo "Precision: $precision"
    echo "Flags: $trtexec_dynamic_flags"
    echo "=========================================================="

    compile_trt_model "fastbev_pre_trt" "$trtexec_dynamic_flags"  1 1 "need"
    compile_trt_model "fastbev_post_trt_decode" "$trtexec_dynamic_flags"  1 3 "noneed"
}

build_target=${1:-single}

if [ "$build_target" == "all" ]; then
    for model_dir in model/*; do
        if [ ! -d "$model_dir" ]; then
            continue
        fi

        model_name=$(basename "$model_dir")
        precision=$(default_precision_for_model "$model_name")
        build_one_model_family "$model_name" "$precision"
    done
else
    build_one_model_family "$DEBUG_MODEL" "$DEBUG_PRECISION"
fi
