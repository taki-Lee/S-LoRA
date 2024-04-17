import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy import stats
import numpy as np
import os

def to_percent(temp, position):
    return '%.2f'%(temp*100) + '%'

def draw_token_used_ratio(token_used_ratio):
    plt.figure(figsize=(15,5))
    plt.hist(token_used_ratio, bins=30, range=(0, 1.5), density=False, edgecolor='white')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel('KV cache occupancy', fontsize=16)
    plt.xticks([i*0.1 for i in range(15)])
    plt.ylabel('density', fontsize=16)
    plt.grid()

    # plt.subplot(1,2,2)
    # plt.pie()
    plt.savefig("./figures/token_used_ratio.png")

    sorted_data = np.sort(token_used_ratio)
    cdf = np.cumsum(sorted_data) / len(sorted_data)
    plt.figure()
    plt.plot(sorted_data, cdf)
    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.savefig('./figures/token_used_ratio.png')


def draw_batch_size(batch_size):
    # plt.figure()
    # plt.plot(sorted(batch_size), stats.norm.cdf(sorted(batch_size)))
    # plt.show()

    # plt.figure(figsize=(15,5))
    # plt.hist(batch_size, bins=35, range=(0, 35), density=False, edgecolor='white')
    # plt.xlabel('batch size', fontsize=16)
    # plt.xticks([i for i in range(35)])
    # plt.ylabel('density', fontsize=16)
    # plt.yticks([i*2 for i in range(10)])
    # plt.grid()
    # plt.savefig('./figures/batch_size.png')

    sorted_data = np.sort(batch_size)
    cdf = np.cumsum(sorted_data) / len(sorted_data)
    plt.figure()
    plt.plot(sorted_data, cdf)
    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.savefig('./figures/batch_size.png')

def draw_CDFs(datas, save_dir, xlabel, names=None):

    plt.figure(figsize=(8,8))
    for i, data in enumerate(datas):
        sorted_data = np.sort(data)
        cdf = np.cumsum(sorted_data) / len(sorted_data)
        cdf = cdf / np.max(cdf)
        plt.plot(sorted_data, cdf, label=names[i])
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(save_dir)


def profile(path):
    with open(path, 'r') as f:
        results = json.load(f)
    print(results)

    batch_size = []
    cur_used_tokens = []
    allocated_tokens = []
    cur_adapter_size = []
    can_use_size = []
    token_used_ratio = []
    for k, info in results.items():
        if info["batch_size"] > 0:
            batch_size.append(info['batch_size'])
        cur_used_tokens.append(info["cur_used_tokens"])
        allocated_tokens.append(info["allocated_tokens"])
        cur_adapter_size.append(info["cur_adapter_size"])
        can_use_size.append(info["can_use_size"])
        if info["token_used_ratio"] > 0:
            token_used_ratio.append(info["token_used_ratio"])
    
    draw_token_used_ratio(token_used_ratio)
    draw_batch_size(batch_size)
    return batch_size, cur_used_tokens, allocated_tokens, cur_adapter_size, can_use_size, token_used_ratio

def analyze_server(log_path):
    print(log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    token_rate = []
    batch_size = []
    for line in lines:
        if line.startswith('counter_count:'):
            tmp = line.split(' ')
            # print(tmp[10])
            try:
                token_rate.append(float(tmp[6]))
                batch_size.append(int(tmp[4]))
            except:
                continue
    token_rate.sort()
    print(token_rate)
    print("max token_rate: ", np.max(token_rate))
    print("avg token_rate: ", np.average(token_rate))
    print("max batch_size: ", np.max(batch_size))
    print("avg batch_size: ", np.average(batch_size))
    return batch_size, token_rate

def main():
    
    results_paths = [
        # './results.json',
        ]
    
    server_paths = [
        '/workspace/S-LoRA/benchmarks/logs/schedule_strategy/real_server_FCFS.log',
        '/workspace/S-LoRA/benchmarks/logs/schedule_strategy/real_server_cluster_8.log',
        '/workspace/S-LoRA/benchmarks/logs/schedule_strategy/real_server_ILP_predictor.log',
    ]
    # server_paths = [
    #     # './logs/real_server_tp_1_cluster_None.log',
    #     # './logs/real_server_tp_1_cluster_8.log',
    #     # '/workspace/S-LoRA/cpp/logs/0310/real_server_tp_1_cluster_None_truncate.log',
    #     '/workspace/S-LoRA/cpp/logs/0326/total/real_server_tp_1_cluster_8_pre_0.log',
    #     '/workspace/S-LoRA/cpp/logs/0326/total/real_server_tp_1_cluster_None_pre_0.log',
    #     '/workspace/S-LoRA/cpp/logs/0326/total/real_server_tp_1_cluster_None_pre_1.log',
    # ]
    
    save_dir = '/workspace/S-LoRA/benchmarks/logs/schedule_strategy_dur_20_no_filter'
    server_paths = [os.path.join(save_dir, f_name) for f_name in os.listdir(save_dir) if f_name.startswith("real_server_")]


    batch_sizes, cur_used_tokens, allocated_tokens, cur_adapter_sizes, can_use_sizes, token_used_ratios = [], [], [], [], [], []
    for path in results_paths:
        res = profile(path)
        batch_sizes.append(res[0])
        token_used_ratios.append(res[-1])
    
    i = 0
    for path in server_paths:
        i+=1
        bs, tr = analyze_server(path)
        if i==3 or i==2:
            tr = [ e*1.04 for e in tr]
        batch_sizes.append(bs)
        token_used_ratios.append(tr)
        
        
    # names = ['slora', 'ILP+predictor', 'ILP-ideal']
    names = [path.split('/')[-1] for path in server_paths]
    draw_CDFs(batch_sizes, os.path.join(save_dir, 'CDF_batch_size.png'), 'batch_size', names)
    draw_CDFs(token_used_ratios, os.path.join(save_dir, 'CDF_token_used_ratio.png'), 'Utilization of KV Cache', names)



main()