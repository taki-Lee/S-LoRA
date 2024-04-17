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
    # logs_dir="logs/schedule_strategy_dur_20_no_filter"
    logs_dir="logs/tests/"
    # ### FCFS
    # echo "schedule requests with FCFS"
    # use_predictor="0"
    # cluster_num='None'
    # use_scheduler="0"
    # export USE_PREDICTOR=${use_predictor}
    # export USE_SCHEDULER=${use_scheduler}
    # export SCHEDULE_FCFS="1"
    # cmd=$(get_cmd)
    # echo $cmd
    # $cmd &> ${logs_dir}/real_server_FCFS.log

    ### LCFS
    # echo "schedule requests with LCFS"
    # use_predictor="0"
    # cluster_num='None'
    # use_scheduler="0"
    # export USE_PREDICTOR=${use_predictor}
    # export USE_SCHEDULER=${use_scheduler}
    # export SCHEDULE_LCFS="1"
    # cmd=$(get_cmd)
    # echo $cmd
    # $cmd &> ${logs_dir}/real_server_LCFS.log

    ### slora with cluster-8
    # echo "schedule requests with slora cluster-8"
    # use_predictor="0"
    # cluster_num='8'
    # export USE_PREDICTOR=${use_predictor}
    # export USE_SCHEDULER=${use_scheduler}
    # cmd=$(get_cmd)
    # echo $cmd
    # $cmd &> ${logs_dir}/real_server_cluster_8.log


    # ### ILP + predictor
    echo "schedule requests with ILP + predictor"
    use_predictor="1"
    cluster_num='None'
    use_scheduler="1"
    export USE_PREDICTOR=${use_predictor}
    export USE_SCHEDULER=${use_scheduler}
    cmd=$(get_cmd)
    echo $cmd
    $cmd &> ${logs_dir}/real_server_ILP_predictor.log
}

test_max_new_token()
{
    logs_dir="logs/max_new_token"

    ### FCFS
    # echo "schedule requests with FCFS"
    # use_predictor="0"
    # cluster_num='None'
    # use_scheduler="0"
    # export USE_PREDICTOR=${use_predictor}
    # export USE_SCHEDULER=${use_scheduler}
    # export SCHEDULE_FCFS="1"
    # cmd=$(get_cmd)
    # echo $cmd
    # $cmd &> ${logs_dir}/real_server_FCFS.log

    ### LCFS
    # echo "schedule requests with LCFS"
    # use_predictor="0"
    # cluster_num='None'
    # use_scheduler="0"
    # export USE_PREDICTOR=${use_predictor}
    # export USE_SCHEDULER=${use_scheduler}
    # export SCHEDULE_LCFS="1"
    # cmd=$(get_cmd)
    # echo $cmd
    # $cmd &> ${logs_dir}/real_server_LCFS.log

    ### slora with cluster-8
    # echo "schedule requests with slora cluster-8"
    # use_predictor="0"
    # cluster_num='8'
    # export USE_PREDICTOR=${use_predictor}
    # export USE_SCHEDULER=${use_scheduler}
    # cmd=$(get_cmd)
    # echo $cmd
    # $cmd &> ${logs_dir}/real_server_cluster_8.log


    # ### ILP + predictor
    echo "schedule requests with ILP + predictor"
    use_predictor="1"
    cluster_num='None'
    use_scheduler="1"
    export USE_PREDICTOR=${use_predictor}
    export USE_SCHEDULER=${use_scheduler}
    cmd=$(get_cmd)
    echo $cmd
    $cmd &> ${logs_dir}/real_server_ILP_predictor.log
}



stratege="FCFS"
# test_schedule_strategy
# test_max_new_token
