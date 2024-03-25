import numpy as np
import os
import json
import argparse

def profile(json_path):
    def cal_avg_output_token_time(v):
        return (v['request_latency'] - v['first_token_latency']) / v['output_len']

    with open(json_path, 'r') as f:
        info = json.loads(f.read())
    output_per_token_time = []
    for k, v in info.items():
        output_per_token_time.append(cal_avg_output_token_time(v))
    
    print(output_per_token_time)
    print("avg per output token time: %f , Var: %f" % (np.average(output_per_token_time), np.var(output_per_token_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default='./real_run_1.json')
    args = parser.parse_args()

    profile(args.json_path)