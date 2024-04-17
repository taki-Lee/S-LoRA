import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from .ILP_scheduler import ILPScheduler
import time


class ILPReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.is_new_req_come = False
        
    def append(self, req):
        self.waiting_req_list.append(req)
        self.is_new_req_come = True
        return
    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                # self.cache_len_list.append((req.input_len + len(req.output_ids),
                #                            req.max_output_len - len(req.output_ids) - 1))
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_new_token - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        # self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.append((req.input_len + 1, req.max_new_token - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False

    # @calculate_time(show=True, min_cost_ms=0.01)
    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        
        # removing aborted request
        self.waiting_req_list = [req for req in self.waiting_req_list if req.aborted == False]
        # print("length of waiting_req_list: ", len(self.waiting_req_list))
        # start_time = time.time()
        best_run_list = self.generate_best_run_list(current_batch, lora_ranks)
        # print("ILP cost time: %f" % ((time.time()-start_time)*1000))
        if len(best_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, best_run_list)
            L = len(self.waiting_req_list)
            self.waiting_req_list = [req for req in self.waiting_req_list if req not in best_run_list]
            # print("removing %d from waiting_req_list, %d requests waiting. " % (L-len(self.waiting_req_list), len(self.waiting_req_list)))
            return new_batch
        else:
            return None

        
        
    def generate_best_run_list(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if len(self.waiting_req_list) == 0:
            # print("waiting req list is empty, no request to serve. skip generating new batch")
            return []
        serving_reqs = current_batch.reqs if current_batch is not None else []
        waiting_reqs = self.waiting_req_list

        solver = ILPScheduler(serving_req_list=serving_reqs, 
                              waiting_req_list=waiting_reqs, 
                              lora_ranks=lora_ranks,
                              max_total_tokens=self.max_total_tokens)
        
        best_run_list = solver.solve()
        return best_run_list


    def next_batch(self):
        # use in prefetch
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None
