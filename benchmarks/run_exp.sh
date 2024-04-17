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

# suite="4090-req-rate"
# logs_dir="logs/schedule_strategy_dur_20_no_filter"
# python run_exp.py --my-exp \
#     --suite ${suite} \
#     --model-setting Real \
#     --mode real \
#     --set-max-new-token \
#     &> ${logs_dir}/real_run_${suite}.log

suite="4090-req-rate"
logs_dir="logs/ILP_effectiveness"
python run_exp.py --my-exp \
    --suite ${suite} \
    --model-setting Real \
    --mode real \
    --set-max-new-token \
    &> ${logs_dir}/real_run_${suite}.log