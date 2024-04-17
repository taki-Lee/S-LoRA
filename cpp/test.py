import numpy as np
import json
import os
from pprint import pprint

def test_idx_count():
    with open('/workspace/S-LoRA/benchmarks/logs/real_server_tp_1_cluster_None_info.log', 'r') as f:
        lines = f.readlines()

    idx_count = [0]*200

    for line in lines:
        if line.startswith("indices"):
            print(line)
            list_str = line.split(":  ")[-1]
            list_str = list_str[1:-2]
            if list_str.split(', ') == ['']:
                continue
            list = [int(s) for s in list_str.split(', ')]
            for i in list:
                idx_count[i] += 1

    for i, v in enumerate(idx_count):
        print("%d -- %d -- %.3f"%(i, v, float(np.sum(idx_count[:i+1]))/np.sum(idx_count)))

def test_ILP_time():
    with open('/workspace/S-LoRA/benchmarks/logs/real_server_tp_1_cluster_None_pre_1.log', 'r') as f:
        lines = f.readlines()
    
    ILP_cost_time = 0

    for line in lines:
        if line.startswith("ILP cost time"):
            float_str = line.split(": ")[-1]
            ILP_cost_time += float(float_str)
    
    print("total ILP cost time: %f ms" % ILP_cost_time)
            
def test_profile_predictor_accuracy():
    path = '/workspace/S-LoRA/cpp/logs/0319/mysuite/real_server_tp_1_cluster_None_pre.log'
    with open(path, 'r') as f:
        lines = f.readlines()
    
    max_output_len = 0
    max_new_len = 0
    predict = 0
    eos = 0

    for line in lines:
        if line.startswith('req abort'):
            if line.endswith('max_output_len\n'):
                max_output_len += 1
                print('sssss')
            elif line.endswith('pre_dict_output_len\n'):
                predict += 1
            elif line.endswith('max_new_len\n'):
                max_new_len += 1

            elif line.endswith('eos\n'):
                eos += 1
    
    acc = (float(max_new_len) + max_output_len + eos) / (max_new_len + max_output_len + predict + eos)
    print("acc of {predict_len < max_output_len} : ", acc)

def test_dataset_len():
    train_dataset_path = '/workspace/distill-bert/lora-inference/my_datasets/extended_mixed_dataset.json'
    serve_dataset_path = '/workspace/S-LoRA/benchmarks/real_trace/my_traces.json'

    with open(train_dataset_path, 'r') as f:
        train_data = json.load(f)
    print("train_dataset_len:", len(train_data))
    
    with open(serve_dataset_path, 'r') as f:
        serve_data = json.load(f)
    print("serve_dataset_len:", len(serve_data))

def collect_overhead():
    print("==================collect overhead==================")
    
    cost = {}

    path = '/workspace/S-LoRA/benchmarks/logs/overhead/real_server_tp_1_cluster_None_pre_1_time.log'
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Function'):
            tmp = line.split(' ')
            func_name = tmp[1]
            time = float(tmp[3])
            if time < 1.0:
                continue
            if func_name not in cost:
                cost[func_name] = 0
                cost[func_name + '_count'] = 0
            cost[func_name] += time
            cost[func_name + '_count'] += 1
    
    print(cost)


def collect_run_exp(direction):
    def collect_info(f_path, info_str):
        res = []
        with open(f_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith(info_str):
                if info_str == 'Average satisfaction:':
                    tmp = float(line.split(' ')[-1])
                else:
                    tmp = float(line.split(' ')[-2])
                res.append(tmp)
        
        return res
                
        
    file_names = [f_name for f_name in os.listdir(direction) if f_name.startswith("real_run_")]
    file_paths = [os.path.join(direction, f_name) for f_name in file_names]
    print(file_paths)
    collect_var = ['Total time:', 'Throughput:', 'Average latency:', 'Average satisfaction:']
    results = {}

    for var in collect_var:
        results[var] = {}
        for f_name, f_path in zip(file_names, file_paths):
            res = collect_info(f_path, var)    
            results[var][f_name] = res
    pprint(results)
    with open(os.path.join(direction, 'results_collect_run_exp.json'), 'w') as f:
        json.dump(results, f)

# test_idx_count()
# test_ILP_time()
# test_profile_predictor_accuracy()
# test_dataset_len()
# collect_overhead()
collect_run_exp("/workspace/S-LoRA/benchmarks/logs/max_new_token")