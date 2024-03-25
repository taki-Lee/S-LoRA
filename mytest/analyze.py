import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import numpy as np

def to_percent(temp, position):
    return '%.2f'%(temp*100) + '%'

def analyze_run_exp(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[8:-14]

    # print(lines[0])
    # print(lines[-1])
    for line in lines:
        if line.startswith('req_id'):
            tmp = line.split(' ')
            req_id = int(tmp[0])
        else:
            pass

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
                token_rate.append(float(tmp[10]))
                batch_size.append(int(tmp[6]))
            except:
                continue
    token_rate.sort()
    print(token_rate)
    print("max token_rate: ", np.max(token_rate))
    plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    
    plt.hist(token_rate, bins=20, range=(0, 1), density=False, edgecolor='white')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel('KV cache occupancy', fontsize=16)
    plt.xticks([i*0.1 for i in range(11)])
    plt.ylabel('density', fontsize=16)
    plt.grid()

    # plt.subplot(1,2,2)
    # plt.pie()
    plt.savefig("./figures/analyze.png")
    plt.show()

    # draw batch_size
    plt.figure(figsize=(15,5))
    plt.hist(batch_size, bins=35, range=(0, 35), density=False, edgecolor='white')
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel('batch size', fontsize=16)
    plt.xticks([i for i in range(35)])
    plt.ylabel('density', fontsize=16)
    plt.yticks([i*2 for i in range(10)])
    plt.grid()
    plt.savefig('./figures/batch_size.png')

# run_exp_path = '/workspace/S-LoRA/benchmarks/logs/real_run_4.log'
# analyze_run_exp(run_exp_path)

# server_path = '../benchmarks/logs/real_server_15.log'
server_path = '../benchmarks/logs/tokens_compare/real_server_1000.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_1000.png')

server_path = '../benchmarks/logs/tokens_compare/real_server_2000.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_2000.png')

server_path = '../benchmarks/logs/tokens_compare/real_server_3072.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_3072.png')

server_path = '../benchmarks/logs/tokens_compare/real_server_4000.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_4000.png')

server_path = '../benchmarks/logs/tokens_compare/real_server_4500.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_4500.png')

server_path = '../benchmarks/logs/tokens_compare/real_server_5000.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_5000.png')


server_path = '/workspace/S-LoRA/cpp/real_server_tp_1_10.log'
analyze_server(server_path)
os.system('cp ./figures/analyze.png ./figures/analyze_10.png')