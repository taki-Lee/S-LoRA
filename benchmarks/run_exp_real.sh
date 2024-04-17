#! /bin/bash

suite='my-suite'
# suite='rate-1'
# suite='rate-2'
# suite='rate-3'
# suite='rate-4'
# suite='rate-5'
# suite='rate-6'

# python run_exp.py --debug \
#     --suite ${suite} \
#     --model-setting Real \
#     --mode real \
#     --set-max-new-token \
#     &> logs/real_run_${suite}.log
# --mode real


test_real_benchmarks()
{
    suites=("real-2" "real-4" "real-6" "real-10" "real-20" "real-30")
    for su in ${suites[*]}
    do
        echo ${su}
        python run_exp.py --debug \
            --suite ${su} \
            --model-setting Real \
            --mode real \
            --set-max-new-token \
            &> logs/real_mode/real_run_${su}.log
    done
}

test_real_benchmarks