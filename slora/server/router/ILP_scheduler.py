import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from slora.utils.infer_utils import use_predictor

class ILPScheduler:
    def __init__(self, serving_req_list, waiting_req_list, lora_ranks, max_total_tokens, is_truncate=True, is_priority=True):
        self.serving_req_list: List[Req] = serving_req_list
        if is_truncate:
            self.waiting_req_list: List[Req] = waiting_req_list[:64]
        else:
            self.waiting_req_list: List[Req] = waiting_req_list

        self.lora_ranks = lora_ranks # adapter size = lora_rank * 4
        self.max_total_tokens = max_total_tokens
        self.is_truncate = is_truncate
        self.is_priority = is_priority
        self.MAX_TIME = 200.0
        self.TRUNCATE_POS = 50

    def SCIP_solve(self):
        # solver = pywraplp.Solver.CreateSolver("SCIP")
        solver = pywraplp.Solver.CreateSolver("SCIP")
        x = []
        y = []
        adapter_count = {}
        waiting_req_num = len(self.waiting_req_list)

        # add x[i]
        for i in range(waiting_req_num):
            x.append(solver.IntVar(0.0, 1.0, 'x_%d'%i))
        
        # add y[i]
        for i, (adp_dir, rank) in enumerate(self.lora_ranks.items()):
            adapter_count[adp_dir] = []
            y.append(solver.IntVar(0.0, 1.0, 'y_%d' % i))

        # serving_req's adapter must in GPU
        for req in self.serving_req_list:
            adapter_count[req.adapter_dir].append(1)
        
        # s.t. for adapter is used
        for i, req in enumerate(self.waiting_req_list):
            adapter_count[req.adapter_dir].append(x[i])
        
        for i, (adp_name, counts) in enumerate(adapter_count.items()):
            solver.Add(y[i] <= solver.Sum(counts))
            # print("counts, ", counts)
            for is_use in counts:
                solver.Add(y[i] >= is_use)
        
        # total adapter size
        adapter_sizes = [y[i] * rank * 4 for i, (adp_name, rank) in enumerate(self.lora_ranks.items())]

        # max needed generate tokens
        cache_list = []
        
        if use_predictor():
            for req in self.serving_req_list:
                has_generate_token = len(req.output_ids)
                cache_list.append((req.input_len+has_generate_token,
                                req.predict_output_len-has_generate_token-1,
                                1))
            
            for i, req in enumerate(self.waiting_req_list):
                cache_list.append((req.input_len+1,
                                req.predict_output_len-1,
                                x[i]))
        else:
            for req in self.serving_req_list:
                has_generate_token = len(req.output_ids)
                cache_list.append((req.input_len+has_generate_token,
                                req.max_output_len-has_generate_token-1,
                                1))
            
            for i, req in enumerate(self.waiting_req_list):
                cache_list.append((req.input_len+1,
                                req.max_output_len-1,
                                x[i]))
        
            
        cache_list.sort(key=lambda x: -x[1])
        
        has_gen_tokens = [e[0] for e in cache_list]
        left_tokens = [e[1] for e in cache_list]
        is_use = [e[2] for e in cache_list]
        # print("has_gen_tokens:", has_gen_tokens)
        # print("left_tokens:", left_tokens)
        size_array = [solver.Sum(is_use[:i+1]) for i in range(len(is_use))]

        used_gen_tokens = [a*b for a,b in zip(has_gen_tokens, is_use)]

        for i in range(len(cache_list)):
            solver.Add(solver.Sum(used_gen_tokens[:i+1]) + left_tokens[i]*size_array[i] + solver.Sum(adapter_sizes) <= self.max_total_tokens)

        solver.Maximize(solver.Sum(x))
        status = solver.Solve()

        best_run_list = []

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        # if status == pywraplp.Solver.OPTIMAL:
        #     print('Objective value =', solver.Objective().Value())
        #     for i in range(req_num):
        #         print(x[i].name(), ' = ', x[i].solution_value())
        #     print()
            # print('Problem solved in %f milliseconds' % solver.wall_time())
        #     print('Problem solved in %d iterations' % solver.iterations())
        #     print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
            for i in range(waiting_req_num):
                if int(x[i].solution_value()) == 1:
                    best_run_list.append(self.waiting_req_list[i])

            # print("serve requests: ", [req.request_id for req in best_run_list])
            # print("indices of best_run_list: ", [self.waiting_req_list.index(req) for req in best_run_list])
        # else:
        #     print('The problem does not have an optimal solution.')
        return best_run_list

    def solve(self):
        if self.is_priority:
            # sort by priority
            sort_list = [(req, float((req.input_len+req.max_output_len)/(self.MAX_TIME - i))) 
                         for i, req in enumerate(self.waiting_req_list)]
            sort_list.sort(key=lambda x:x[1])
            self.waiting_req_list = [req for (req, pri) in sort_list]
            
        if self.is_truncate:
            # truncate
            self.waiting_req_list = self.waiting_req_list[:self.TRUNCATE_POS]
        
        if True:
            best_run_list = self.SCIP_solve()
        else:
            # slower
            best_run_list = self.CP_SAT_solve()

        return best_run_list
        
