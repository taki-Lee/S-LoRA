#! /bin/bash

suite='my-suite'
# suite='rate-1'
# suite='rate-2'
# suite='rate-3'
# suite='rate-4'
# suite='rate-5'
# suite='rate-6'

# suite="real-2"
# suite="real-3"
# suite="real-4"
# suite="real-5"
# suite="real-6"
# suite="real-7"
# suite="real-8"
# suite="real-10"

# python run_exp.py --debug \
#     --suite ${suite} \
#     --model-setting Real \
#     --mode real \
#     --set-max-new-token \
#     &> logs/real_run_${suite}.log



# suite="a100-req-rate"
# # logs_dir="logs/ILP_effectiveness"
# logs_dir="logs/req_rate"
# model_setting="debug-13b"
# python run_exp.py --my-exp \
#     --suite ${suite} \
#     --model-setting ${model_setting} \
#     --mode real \
#     --set-max-new-token \
#     &> ${logs_dir}/real_run_${suite}.log

# suite="a100-max-new-token"
# logs_dir="logs/max_new_token"

# suite="a100-num-adapter"
# logs_dir="logs/num_adapters"

# suite="a100-req-rate"
# logs_dir="logs/ILP_effectiveness"

# suite="a100-overhead"
# suite="a100-discover-problem"
# logs_dir="logs/analyze_memory"

suite="a100-req-rate"
logs_dir="logs/test"

model_setting="debug-13b"
python run_exp.py --my-exp \
    --suite ${suite} \
    --model-setting ${model_setting} \
    --mode real \
    --set-max-new-token \
    &> ${logs_dir}/real_run_${suite}.log