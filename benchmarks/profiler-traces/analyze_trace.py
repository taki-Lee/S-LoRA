import json

def profile_trace(trace_path):

    with open(trace_path, 'r') as f:
        trace = json.load(f)
    events = trace['traceEvents']
    print(trace_path, ' ', len(events))

    times = {}
    count = {}

    for e in events:
        if 'dur' not in e:
            continue
        if e['name'] in times:
            times[e['name']] += int(e['dur'])
            count[e['name']] += 1
        else:
            times[e['name']] = int(e['dur'])
            count[e['name']] = 1
    
    return times, count
        

def analyze(cpu_trace_path, cuda_trace_path):
    cuda_times, cuda_count = profile_trace(cuda_trace_path)
    cpu_times, cpu_count = profile_trace(cpu_trace_path)

    print(cuda_times)
    print(cpu_times)

    cpu_keys = set(cpu_times.keys())
    cuda_keys = set(cuda_times.keys())

    commom_keys = cpu_keys & cuda_keys

    # writting 
    with open('annalyze_result.log', 'w') as f:
        comm_info = {}
        for k in commom_keys:
            tmp = {}
            tmp['cpu time'] = cpu_times[k]
            tmp['cuda time '] = cuda_times[k]
            tmp['cpu count'] = cpu_count[k]
            tmp['cuda count'] = cuda_count[k]
            comm_info[k] = tmp
            if cpu_times[k]/cpu_count[k] < cuda_times[k]/cuda_count[k]:
                f.write("average of %s exec time:  %.2f us in cpu -- %.2f us in cuda\n" % (k, cpu_times[k]/cpu_count[k], cuda_times[k]/cuda_count[k]))
        with open('comm_times.json', 'w') as fp:
            json.dump(comm_info, fp)

        f.write("cpu_time only\n")
        for k, v in cpu_times.items():
            if k not in commom_keys:
                f.write("%s: %d us\n" % (k, cpu_times[k]))

        f.write("cuda_time only")
        for k, v in cuda_times.items():
            if k not in commom_keys:
                f.write("%s: %d us\n" % (k, cuda_times[k]))

    with open('cpu_times.json', 'w') as f:
        json.dump(cpu_times, f)
    with open('cuda_times.json', 'w') as f:
        json.dump(cuda_times, f)


if __name__ == '__main__':
    cpu_trace_path = 'trace-prefill.json'
    cuda_trace_path = 'trace-decode.json'
    analyze(cpu_trace_path, cuda_trace_path)