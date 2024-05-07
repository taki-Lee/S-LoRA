import json
import ortools
from ortools.linear_solver import pywraplp
import os
import time
import numpy as np

MAX_TIME = 4000
ADAPTER_NAME2SIZE = {}
RESERVE_MEM_RATIO = 0.0

def load_info(path):
    arr_time = [0] * 1010
    prompt_len = [0] * 1010
    output_len = [0] * 1010
    first_token_time = [0] * 1010
    lora_dir = [0] * 1010
    lora_size = [0] * 1010
    arrive_reqs = []
    for i in range(MAX_TIME):
        arrive_reqs.append([])
    # arrive_reqs = [[]]*MAX_TIME

    with open(path, 'r') as f:
        lines = f.readlines()

    tmp = lines[0].split(' ')
    global req_num, total_token_num
    req_num = int(tmp[0])
    total_token_num = int(tmp[1])
    
    for line in lines[1:]:
        tmp = line.split(' ')
        req_id = int(tmp[0])
        arr_time[req_id], prompt_len[req_id], output_len[req_id], first_token_time[req_id], lora_dir[req_id], lora_size[req_id] = (
            int(float(tmp[1]) * 1000 / 30), int(tmp[2]), int(tmp[3]), float(tmp[4]), tmp[5], int(tmp[6])
        )
        arrive_reqs[arr_time[req_id]].append(req_id)
        ADAPTER_NAME2SIZE[lora_dir[req_id]] = lora_size[req_id]
    return arr_time, prompt_len, output_len, first_token_time, lora_dir, lora_size, arrive_reqs
    
def offload_req(req_id, prompt_len, output_len, cur_tokens):
    # 
    cur_tokens = cur_tokens - (prompt_len[req_id] + output_len[req_id])

    # TODO: offload adapter

def cal_used_tokens(serving_reqs, prompt_len, start_time, cur_time):
    token = 0
    for req_id in serving_reqs:
        token += prompt_len[req_id] + cur_time - start_time[req_id]
    return token

def cal_adapter_size(serving_reqs, lora_dir):
    serving_adapters = set()
    adapter_size = 0
    for req_id in serving_reqs:
        if lora_dir[req_id] not in serving_adapters:
            serving_adapters.add(lora_dir[req_id])
            adapter_size += ADAPTER_NAME2SIZE[lora_dir[req_id]]
            print(lora_dir[req_id], ADAPTER_NAME2SIZE[lora_dir[req_id]])

    return adapter_size

def cal_can_use_mem(serving_reqs, prompt_len, req_start_time, cur_time, lora_dir):
    serving_adapters = set()
    serving_tokens = 0
    adapter_size = 0
    for req_id in serving_reqs:
        serving_tokens += prompt_len[req_id] + cur_time - req_start_time[req_id]
        if lora_dir[req_id] not in serving_adapters:
            serving_adapters.add(lora_dir[req_id])
            adapter_size += ADAPTER_NAME2SIZE[lora_dir[req_id]]
    
    return total_token_num - serving_tokens - adapter_size
    
def cal_max_tokens(serving_reqs, prompt_len, output_len):
    max_tokens = 0
    for req_id in serving_reqs:
        max_tokens += prompt_len[req_id] + output_len[req_id]
    return max_tokens

def cal_max_use_tokens(serving_reqs, prompt_len, output_len, start_time, cur_time):
    cache_list = []
    for req_id in serving_reqs:
        has_gen_token = cur_time - start_time[req_id]
        cache_list.append((prompt_len[req_id]+has_gen_token, output_len[req_id]-has_gen_token))
    
    cache_list.sort(key=lambda x:-x[1])
    has_gen_tokens = [e[0] for e in cache_list]
    left_tokens = np.array([e[1] for e in cache_list])
    size_array = np.arange(1, len(cache_list) + 1, 1)

    cum_run_len_array = np.cumsum(has_gen_tokens)
    
    need_max_token_num = (left_tokens * size_array + cum_run_len_array).max()

    return need_max_token_num


def ILP(request_pool, serving_reqs, prompt_len, output_len, lora_dir, req_start_time, cur_time):
    req_num = len(request_pool)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = []
    adapter_count = {}

    # Add x[i]
    for i in range(req_num):
        x.append(solver.IntVar(0.0, 1.0, 'x_%d' % i))

    # Add y[i]
    y = []
    for i, (adp_name, size) in enumerate(ADAPTER_NAME2SIZE.items()):
        adapter_count[adp_name] = []
        y.append(solver.IntVar(0.0, 1.0, 'y_%d' % i))

    for req_id in serving_reqs:
        adp_name = lora_dir[req_id]
        adapter_count[adp_name].append(1)
    
    for i, req_id in enumerate(request_pool):
        adp_name = lora_dir[req_id]
        adapter_count[adp_name].append(x[i])

    for i, (adp_name, counts) in enumerate(adapter_count.items()):
        solver.Add(y[i] <= solver.Sum(counts))
        print("counts, ", counts)
        for is_use in counts:
            solver.Add(y[i] >= is_use)
    
    adapter_sizes = [ y[i] * size for i, (adp_name, size) in enumerate(ADAPTER_NAME2SIZE.items())]
    
    USE_SIMPLE=False
    if USE_SIMPLE:
        # simple Memory s.t.
        serving_tokens = 0
        for req_id in serving_reqs:
            serving_tokens += prompt_len[req_id] + output_len[req_id]
        # serving_tokens = cal_used_tokens(serving_reqs, prompt_len, req_start_time, cur_time)

        token_to_serve = [x[i]*(prompt_len[i] + output_len[i]) for i in range(req_num)]
        solver.Add(solver.Sum(token_to_serve) + solver.Sum(adapter_sizes) + serving_tokens <= total_token_num * (1 - RESERVE_MEM_RATIO))
    else:
        # accurate memory s.t.
        cache_list = []
        # add serving req {serving_len, left_len, 1} to cache_list
        for req_id in serving_reqs:
            has_generate_token = cur_time - req_start_time[req_id]
            cache_list.append((prompt_len[req_id] + has_generate_token, output_len[req_id]-has_generate_token, 1))
        # add to_serve req {prompt_len, output_len, x[i](i=0,1,...,len(req_pool)-1)} to cache_list
        for i, req_id in enumerate(request_pool):
            cache_list.append((prompt_len[req_id], output_len[req_id], x[i]))
        # sort cache_list from large to small according output_len
        cache_list.sort(key=lambda t: -t[1])
        
        has_gen_tokens = [e[0] for e in cache_list]
        left_tokens = [e[1] for e in cache_list]
        is_use = [e[2] for e in cache_list]

        # size_array
        size_array = [solver.Sum(is_use[:i+1]) for i in range(len(is_use))]
        
        print("req has_gen_tokens:", has_gen_tokens)
        print("is_use", is_use)
        # used_gen_tokens = has_gen_tokens*is_use
        used_gen_tokens = [a*b for a,b in zip(has_gen_tokens, is_use)]
        
        # Add Memory s.t.
        assert len(cache_list) == len(size_array), "the length of size_array must equal with cache_list"
        # for i, (gen_tokens, size) in enumerate(zip(has_gen_tokens, size_array)):
        #     solver.Add(gen_tokens + size_array*left_tokens[i] <= total_token_num)
        for i in range(len(cache_list)):
            solver.Add(solver.Sum(used_gen_tokens[:i+1]) + left_tokens[i]*size_array[i] + solver.Sum(adapter_sizes) <= total_token_num)


    solver.Maximize(solver.Sum(x))
    
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
    #     print('Objective value =', solver.Objective().Value())
    #     for i in range(req_num):
    #         print(x[i].name(), ' = ', x[i].solution_value())
    #     print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
    #     print('Problem solved in %d iterations' % solver.iterations())
    #     print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
    # else:
    #     print('The problem does not have an optimal solution.')
    
    reqs = []
    for i in range(req_num):
        if int(x[i].solution_value()) == 1:
            reqs.append(request_pool[i])

    # assert cal_max_use_tokens(serving_reqs + reqs, prompt_len, output_len, req_start_time, cur_time) \
    #         + cal_adapter_size(serving_reqs + reqs, lora_dir) \
    #         <= total_token_num \
    #         , "need to satisfy {} + {} <= {}".format(cal_max_use_tokens(serving_reqs + reqs, prompt_len, output_len, req_start_time, cur_time), 
    #                                                  cal_adapter_size(serving_reqs + reqs, lora_dir),
    #                                                  total_token_num) 

    return reqs


def main():
    info_path = './info_10.txt'
    os.system('rm -rf results.json')
    results = {}
    arr_time, prompt_len, output_len, first_token_time, lora_dir, lora_size, arrive_reqs = load_info(info_path)

    request_pool = []
    serving_reqs = []
    serving_adapters = []
    cur_tokens = 0
    start_time = [0] * 1010

    end_reqs = [] # t时刻, 哪些请求服务完成
    for i in range(MAX_TIME):
        end_reqs.append([])
    
    can_use_size = total_token_num

    for cur_time in range(MAX_TIME):
        print("======== cur_time =", cur_time, "========")
        if len(arrive_reqs[cur_time]) != 0 or len(end_reqs[cur_time])!=0:
            # 1. Check if there is a request coming at t
            request_pool.extend(arrive_reqs[cur_time])
            print("requests ", arrive_reqs[cur_time], " arrving at ", cur_time)
            # 2. offload requests and adapters
            print("offloading ", end_reqs[cur_time])
            for req_id in end_reqs[cur_time]:
                serving_reqs.remove(req_id)

            # 3. run algorithm to get serve requests
            alg_start_time = time.time()
            reqs = ILP(request_pool, serving_reqs, prompt_len, output_len, lora_dir, start_time, cur_time)
            print("ILP cost time : ", (time.time() - alg_start_time) * 1000, ' milliseconds')
            # 4. update the end_time of end_reqs
            serving_reqs.extend(reqs)
            for req_id in reqs:
                request_pool.remove(req_id)
                start_time[req_id] = cur_time
                end_reqs[cur_time + output_len[req_id]].append(req_id)

        if cur_time % 50 == 0 and len(serving_reqs) != 0:
            cur_used_tokens = cal_used_tokens(serving_reqs, prompt_len, start_time, cur_time)
            print(" serving reqs:", serving_reqs, '\n',
                    "request pool:", request_pool, '\n',
                    "batch_size:", len(serving_reqs), '\n',
                    "cur_used_tokens:", cur_used_tokens, 
                    "allocated_tokens:", cal_max_tokens(serving_reqs, prompt_len, output_len),
                    "cal_max_use_tokens:", cal_max_use_tokens(serving_reqs, prompt_len, output_len, start_time, cur_time),
                    "cur_adapter_size:", cal_adapter_size(serving_reqs, lora_dir),
                    "can_use_size:", cal_can_use_mem(serving_reqs, prompt_len, start_time, cur_time, lora_dir), 
                    "token_used_ratio:", cur_used_tokens / total_token_num, 
                    )
            
            info = dict()
            info['batch_size'] = len(serving_reqs)
            info["cur_used_tokens"] = cur_used_tokens
            info["allocated_tokens"] = cal_max_tokens(serving_reqs, prompt_len, output_len)
            info["cur_adapter_size"] = cal_adapter_size(serving_reqs, lora_dir)
            info["can_use_size"] = cal_can_use_mem(serving_reqs, prompt_len, start_time, cur_time, lora_dir)
            info["token_used_ratio"] = cur_used_tokens / total_token_num

            results[cur_time] = info
                

            # serving_adapter.extend()

    with open('results.json', 'a') as f:
        json.dump(results, f)

main()


