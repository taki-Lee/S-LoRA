import matplotlib.pyplot as plt
import json
import os

def _draw_lines(title, line_names, x_values, y_values, save_dir, **kwargs):
    plt.figure()
    plt.title(title)
    plt.xlabel('request rate')
    plt.ylabel(title) 

    for name, y in zip(line_names, y_values):
        name = name[9:]
        # linestyle
        color = 'red'
        line_style = '-'
        if 'FCFS.log' in name:
            color = 'sandybrown'
            line_style = '--'
        if 'FCFS_predictor.log' in name:
            color = 'darkgoldenrod'
            line_style = '-.'
        if 'slora.log' in name:
            color = 'limegreen'
            line_style = '--'
        if 'slora_predictor.log' in name:
            color = 'darkgreen'
            line_style = '-.'
        if 'ILP_predictor.log' in name:
            color = 'cornflowerblue'
            line_style = '-'
        marker = 'o'
        plt.plot(x_values, y, label=name, linestyle=line_style, marker=marker, color=color)
        plt.xticks(x_values)
        if "xlim" in kwargs:
            plt.xlim(kwargs['xlim'])
    
    plt.legend()
    plt.savefig(save_dir)

def draw_req_rate():
    x_values = [4, 8, 12, 16, 20, 22, 24]
    json_path = "/home/hadoop-hdp/codes/S-LoRA/benchmarks/logs/draw_req_rate/results_collect_run_exp.json"
    with open(json_path, 'r') as f:
        datas = json.load(f)

    for name, d in datas.items():
        line_names = d.keys()
        y_values = d.values()
        save_path = os.path.join(os.path.dirname(json_path), "./figures/" + name + ".png")
        _draw_lines(name, line_names, x_values, y_values, save_path, xlim=(3,25))



def draw_max_token():
    x_values = [64, 128, 256, 512]
    json_path = "/home/hadoop-hdp/codes/S-LoRA/benchmarks/logs/max_new_token/results_collect_run_exp.json"
    with open(json_path, 'r') as f:
        datas = json.load(f)

    for name, d in datas.items():
        line_names = d.keys()
        y_values = d.values()
        save_path = os.path.join(os.path.dirname(json_path), "./figures/" + name + ".png")
        _draw_lines(name, line_names, x_values, y_values, save_path)


def draw_num_adapters():
    x_values = [1, 20, 50, 100]
    json_path = "/home/hadoop-hdp/codes/S-LoRA/benchmarks/logs/num_adapters/results_collect_run_exp.json"
    with open(json_path, 'r') as f:
        datas = json.load(f)

    for name, d in datas.items():
        line_names = d.keys()
        y_values = d.values()
        save_path = os.path.join(os.path.dirname(json_path), "./figures/" + name + ".png")
        _draw_lines(name, line_names, x_values, y_values, save_path)


draw_req_rate()
draw_max_token()
draw_num_adapters()