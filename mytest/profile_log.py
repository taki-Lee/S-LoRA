import numpy as np
import os
import json

def profile_log(log_path):
    def startswith_num(s):
        try:
            a = int(s.split(' ')[0])
            b = float(s.split(' ')[1])
            return True
        except:
            return False
    with open(log_path, 'r') as f:
        lines = f.readlines()
    info = {}
    _results = lines[-14:]
    for line in lines:
        if line.startswith('req_id '):
            tmp = line.split(' ')
            req_info = {}
            req_id = int(tmp[1])
            req_info['prompt_len'] = int(tmp[3])
            req_info['output_len'] = int(tmp[5])
            req_info['request_latency'] = float(tmp[7])
            req_info['first_token_latency'] = float(tmp[10])
            info[req_id].update(req_info)
        elif startswith_num(line):
            tmp = line.split(' ')
            req_id = int(tmp[0])
            req_info = {}
            req_info['request_time'] = float(tmp[1])
            req_info['waiting time'] = float(tmp[3])
            req_info['adapter_dir'] = tmp[4]
            info[req_id] = req_info
    
    return info




log_path = '../benchmarks/logs/real_run.log'
info = profile_log(log_path)
out_name = 'real_run.json'

print(info)
with open(out_name, 'w') as f:
    json.dump(info, f)