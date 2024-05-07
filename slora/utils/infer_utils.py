import torch
import torch.distributed as dist

import time
import os 
import nvtx
import functools
import logging
logging.basicConfig(level=logging.INFO)

from slora.server.io_struct import Req, Batch, BatchAbortReq

is_show_cost_time = False
TORCH_STREAMS = [
    torch.cuda.Stream(),
    torch.cuda.Stream(),
]

def mark_cost_time(func_name):
    def inner_func(func):
        def time_func(*args, **kwargs):
            if dist.get_rank() in [0, 1] and is_show_cost_time:
                torch.cuda.synchronize()
                start_time = time.time()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                print(func_name, "cost time:", (time.time() - start_time) * 1000)
                return ans
            else:
                torch.cuda.synchronize()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                return ans

        return time_func

    return inner_func


time_mark = {}


def mark_start(key):
    torch.cuda.synchronize()
    global time_mark
    time_mark[key] = time.time()
    return


def mark_end(key, print_min_cost=0.0):
    torch.cuda.synchronize()
    global time_mark
    cost_time = (time.time() - time_mark[key]) * 1000
    if cost_time > print_min_cost:
        print(f"cost {key}:", cost_time)


def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.time() - start_time) * 1000
                if cost_time > min_cost_ms:
                    print(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


def set_random_seed(seed: int) -> None:
    import random

    random.seed(seed)
    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_profiler():
    try:
        return os.environ["PROFILER"]
    except Exception as e:
        return None
    
def nvtx_decorator_async(func_name, color="gray"):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with nvtx.annotate(func_name, color=color):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def nvtx_decorator(func_name, color='gray'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # logging.info("func_name: " + func_name)
            with nvtx.annotate(func_name, color=color):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def use_stream():
    return "USE_STREAM" in os.environ and os.environ["USE_STREAM"] == '1'

def stream_wrapper(stream):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with torch.cuda.stream(stream):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def use_predictor():
    if "USE_PREDICTOR" in os.environ and os.environ["USE_PREDICTOR"] == '1':
        return True
    return False

def use_scheduler():
    if "USE_SCHEDULER" in os.environ and os.environ["USE_SCHEDULER"] == '1':
        return True
    return False

def use_FCFS():
    if "SCHEDULE_FCFS" in os.environ and os.environ["SCHEDULE_FCFS"] == '1':
        return True
    return False

def use_LCFS():
    if "SCHEDULE_LCFS" in os.environ and os.environ["SCHEDULE_LCFS"] == '1':
        return True
    return False