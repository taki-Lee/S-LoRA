import json

from sympy import true

ADAPTERS = [
    '/workspace/S-LoRA/LLM-models/LLaMA-2-7b/Adapters/MBZUAI/bactrian-x-llama-7b-lora',
    '/workspace/S-LoRA/LLM-models/LLaMA-2-7b/Adapters/tloen/alpaca-lora-7b',
]
ADAPTER_SIZE = [
    64*4,
    16*4,
]

prompt_len = [0]*1010
output_len = [0]*1010
first_token_latency = [0]*1010
arrive_times = [0]*1010
adapters = [0]*1010

def starts_num(s):
    tmp = s.split(' ')[0]
    try:
        t = int(tmp)
        return True
    except:
        return False

def profile_run(path):
    with open (path, 'r') as f:
        lines = f.readlines()
    print(lines[4])
    global n
    n = int(lines[4].split(': ')[-1])
    
    print("number of request: ", n)
    lines = lines[9:-14]
    
    for line in lines:
        tmp = line.split(' ')
        if line.startswith('req_id'):
            id = int(tmp[1])
            input_tokens = int(tmp[3])
            output_tokens = int(tmp[5])
            first_token_time = float(tmp[-2])
            
            prompt_len[id] = input_tokens
            output_len[id] = output_tokens
            first_token_latency[id] = first_token_time
            
        elif starts_num(line):
            # print(tmp)
            id = int(tmp[0])
            time = float(tmp[1])
            arrive_times[id] = time
            
            adp = tmp[-1][:-1]
            for i, adp_name in enumerate(ADAPTERS):
                if adp.startswith(adp_name):
                    adapters[id] = {
                        'dir': adp,
                        'size': ADAPTER_SIZE[i]
                    }

            
def main():
    profile_run('./real_run_10.log')
    with open("info_10.txt", 'w') as f:
        f.write('%d 10000\n' % (n))
        for i in range(n):

            f.write("%d %f %d %d %f %s %d\n" 
                    % (i, arrive_times[i], prompt_len[i], output_len[i], first_token_latency[i], adapters[i]['dir'], adapters[i]['size']))
    
    

main()

"""
request_num total_tokens_num
req_id arrive_time input_token output_token first_token_latency

"""