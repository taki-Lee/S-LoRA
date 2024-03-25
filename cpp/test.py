import numpy as np

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
    with open('/workspace/S-LoRA/benchmarks/logs/real_server_tp_1_cluster_None.log', 'r') as f:
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

test_idx_count()
test_ILP_time()
test_profile_predictor_accuracy()