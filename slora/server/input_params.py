class ServeParams:

    def __init__(
        self,
        first_slo,
        token_slo,
    ) -> None:
        self.first_slo = first_slo
        self.token_slo = token_slo
        return
    
   
    def to_dict(self):
        ret = {}
        ret["first_slo"] = self.first_slo
        ret["token_slo"] = self.token_slo
        return ret


class InputParams:

    def __init__(
        self,
        max_req_total_len,
        # kv cache manager parameters
        max_total_token_num,
        pool_size_lora,
        batch_max_tokens,
        running_max_req_size,
        # mem_ratio,
        # adapter_ratio,
        # heuristic
        swap,
        prefetch,
        prefetch_size,
        scheduler,
        profile,
        batch_num_adapters,
        enable_abort,
        # kernel,
        # # debug
        dummy,
        no_lora_compute,
        no_lora_swap,
        # no_lora_copy,
        no_kernel,
        no_mem_pool,
        bmm,
    ) -> None:
        self.max_req_total_len = max_req_total_len
        self.max_total_token_num = max_total_token_num
        self.pool_size_lora = pool_size_lora
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # self.mem_ratio = mem_ratio
        # self.adapter_ratio = adapter_ratio

        self.swap = swap
        self.prefetch = prefetch
        self.prefetch_size = prefetch_size
        self.scheduler = scheduler
        self.profile = profile
        self.batch_num_adapters = batch_num_adapters
        self.enable_abort = enable_abort
        # self.kernel = kernel

        self.dummy = dummy
        self.no_lora_compute = no_lora_compute
        self.no_lora_swap = no_lora_swap
        # self.no_lora_copy = no_lora_copy
        self.no_kernel = no_kernel
        self.no_mem_pool = no_mem_pool
        self.bmm = bmm
        return
    
    def __str__(self):

        return "\nmax_req_total_len = " + str(self.max_req_total_len) + \
            "\nmax_total_token_num = " + str(self.max_total_token_num) + \
            "\npool_size_lora =" + str(self.pool_size_lora) + \
            "\nself.batch_max_tokens = " + str(self.batch_max_tokens) + \
            "\nself.running_max_req_size = " + str(self.running_max_req_size) + \
            "\nself.swap =" + str(self.swap) + \
            "\nself.prefetch =" + str(self.prefetch) + \
            "\nself.prefetch_size = "+ str(self.prefetch_size) + \
            "\nself.scheduler = "+ str(self.scheduler) + \
            "\nself.profile =" + str(self.profile) + \
            "\nself.batch_num_adapters =" + str(self.batch_num_adapters) + \
            "\nself.enable_abort =" + str(self.enable_abort) + \
            "\nself.dummy = "+ str(self.dummy) + \
            "\nself.no_lora_compute = "+ str(self.no_lora_compute) + \
            "\nself.no_lora_swap = "+ str(self.no_lora_swap) + \
            "\nself.no_kernel = "+ str(self.no_kernel) + \
            "\nself.no_mem_pool = "+ str(self.no_mem_pool) + \
            "\nself.bmm = "+ str(self.bmm)
        
 
