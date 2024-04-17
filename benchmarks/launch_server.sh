#! /bin/bash

# configs
tensor_parallel="0"

nsys_profile="1"
model_setting="Real"
use_stream='0'


# parallel strategy
if [ $tensor_parallel = "1" ]; then
    CUDA_VISIBLE_DEVICES="0,1"
    tp="2"
else
    CUDA_VISIBLE_DEVICES="0"
    tp="1"
fi

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PROFILER=""
export USE_STREAM="${use_stream}"

cpu_cores="16-23"
req_rate="4"
total_time="20"

get_cmd()
{
    cmd="python launch_server.py \
    --num-adapter 100 \
    --num-token 10000 \
    --tp ${tp} \
    --model-setting  ${model_setting} \
    "
    if [ $cluster_num != "None" ]; then
        cmd="${cmd}"" --batch-num-adapters ${cluster_num}"
    fi

    if [ $use_scheduler = "1" ]; then
        cmd="${cmd}"" --scheduler ILP"
    fi

    nsys_file_name="./slora-server-${tp}-${req_rate}-${total_time}"
    if [ $use_stream = "1" ]; then
        nsys_file_name="${nsys_file_name}-stream"
    fi
    
    cmd="taskset -c ${cpu_cores} "" ${cmd}"

    if [ $nsys_profile = "1" ]; then
        cmd="nsys profile -t cuda,nvtx -s none --show-output=true --gpu-metrics-device=$CUDA_VISIBLE_DEVICES --force-overwrite true \
        -o ${nsys_file_name} "" ${cmd}"
    fi
    echo $cmd
}

test_schedule_strategy()
{
    stra=$1
    logs_direction=$2
    echo $stra
    echo $logs_direction

    case $stra in
        "FCFS")
            echo "schedule requests with FCFS"
            use_predictor="0"
            cluster_num='None'
            use_scheduler="0"
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            export SCHEDULE_FCFS="1"
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_direction}/real_server_${stra}.log
            ;;
        "LCFS")
            echo "schedule requests with LCFS"
            use_predictor="0"
            cluster_num='None'
            use_scheduler="0"
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            export SCHEDULE_LCFS="1"
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_dir}/real_server_${stra}.log
            ;;
        "slora")
            echo "schedule requests with slora cluster-8"
            use_predictor="0"
            cluster_num='8'
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_dir}/real_server_${stra}.log
            ;;
        "FCFS_predictor")
            echo "schedule requests with FCFS_predictor"
            use_predictor="1"
            cluster_num='None'
            use_scheduler="0"
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            export SCHEDULE_FCFS="1"
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_direction}/real_server_${stra}.log
            ;;
        "LCFS_predictor")
            echo "schedule requests with LCFS_predictor"
            use_predictor="1"
            cluster_num='None'
            use_scheduler="0"
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            export SCHEDULE_LCFS="1"
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_dir}/real_server_${stra}.log
            ;;
        "slora_predictor")
            echo "schedule requests with slora_predictor cluster-8"
            use_predictor="1"
            cluster_num='8'
            use_scheduler='0'
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_dir}/real_server_${stra}.log
            ;;
        "ILP_predictor")
            echo "schedule requests with ILP + predictor"
            use_predictor="1"
            cluster_num='None'
            use_scheduler="1"
            export USE_PREDICTOR=${use_predictor}
            export USE_SCHEDULER=${use_scheduler}
            cmd=$(get_cmd)
            echo $cmd
            $cmd &> ${logs_dir}/real_server_${stra}.log
            ;;
        *)
            echo "no match strategy"
            ;;
    esac
}

# strategy must in ["FCFS", "slora", "FCFS_predictor", "slora_predictor", "ILP_predictor",]
# strategy="FCFS"
# logs_dir="logs/ILP_effectiveness"
# test_schedule_strategy $strategy $logs_dir

strategy="FCFS_predictor"
logs_dir="logs/ILP_effectiveness"
test_schedule_strategy $strategy $logs_dir
