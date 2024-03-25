import numpy as np
import os
import json
import matplotlib.pyplot as plt

logs_dir = '../benchmarks/logs/'
logs = os.listdir(logs_dir)

print(logs)

def get_req_rate_from_name(log_name):
    return int(log_name.split('_')[-1].split('.')[0])

def results_to_dict(resutls):
    res = {}
    res['Total time'] = float(results[0].split(': ')[-1].split(" s")[0])
    res['Aborted Request'] = int(results[1].split(': ')[-1].split('\n')[0])
    res['Throughput'] = float(results[2].split(': ')[-1].split('req')[0])
    res['Throughput strip'] = float(results[3].split(': ')[-1].split('req')[0])
    res['Average latency'] = float(results[4].split(': ')[-1].split('s\n')[0])
    res['Average latency per token'] = float(results[5].split(': ')[-1].split('s\n')[0])
    res['Average latency per output token'] = float(results[6].split(': ')[-1].split('s\n')[0])
    res['Average first token latency'] = float(results[7].split(': ')[-1].split('s\n')[0])
    res['90 percentile first token latency'] = float(results[8].split(': < ')[-1].split('s\n')[0])
    res['50 percentile first token latency'] = float(results[9].split(': < ')[-1].split('s\n')[0])
    res['Average satisfaction'] = float(results[10].split(': ')[-1].split('\n')[0])
    res['90 percentile satisfaction'] = float(results[11].split(': > ')[-1].split('\n')[0])
    res['50 percentile satisfaction'] = float(results[12].split(': > ')[-1].split('\n')[0])
    res['Average attainment'] = float(results[13].split(': ')[-1].split('\n')[0])
    return res

def draw(x, y, x_label, y_label, fig_pos):
    plt.subplot(fig_pos[0],fig_pos[1],fig_pos[2])
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)

req_rates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
total_times = [0]*16
throughput = [0]*16
throughput_strip = [0]*16
avg_latency = [0]*16
avg_latency_per_token = [0]* 16
avg_latency_per_output_token = [0]*16
avg_latency_first_token = [0]*16

for log in logs:
    if log.startswith('real_server') or log=='real_run.log':
        continue
    path = os.path.join(logs_dir, log)
    with open(path,'r') as f:
        lines = f.readlines()
    results = lines[-14:]
    req_rate = get_req_rate_from_name(log)
    res = results_to_dict(results)
    print(res)

    total_times[req_rate] = res['Total time']
    throughput[req_rate] = res['Throughput']
    throughput_strip[req_rate] = res['Throughput strip']
    avg_latency[req_rate] = res['Average latency']
    avg_latency_per_token[req_rate] = res['Average latency per token']
    avg_latency_per_output_token[req_rate] = res['Average latency per output token']
    avg_latency_first_token[req_rate] = res['Average first token latency']

# print(req_rates)
plt.figure(figsize=(20,10), dpi=100, facecolor='w')
# plt.subplot(4,2,1)
# plt.xlabel('req_rates')
# plt.ylabel('total_times')
# plt.plot(req_rates, total_times[1:])
draw(req_rates, total_times[1:], 'req_rates', 'total times', (2,4,1))
draw(req_rates, throughput[1:], 'req_rates', 'throughput', (2,4,2))
draw(req_rates, throughput_strip[1:], 'req_rates', 'throughput_strip', (2,4,3))
draw(req_rates, avg_latency[1:], 'req_rates', 'avg_latency', (2,4,4))
draw(req_rates, avg_latency_per_token[1:], 'req_rates', 'avg_latency_per_token', (2,4,5))
draw(req_rates, avg_latency_per_output_token[1:], 'req_rates', 'avg_latency_per_output_token', (2,4,6))
draw(req_rates, avg_latency_first_token[1:], 'req_rates', 'avg_latency_first_token', (2,4,7))


plt.show()
plt.savefig('results.png')