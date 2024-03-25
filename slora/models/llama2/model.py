import os
import json
import torch

from slora.models.llama2.layer_infer.transformer_layer_infer import Llama2TransformerLayerInfer
from slora.models.llama2.layer_weights.transformer_layer_weight import Llama2TransformerLayerWeight

from slora.models.llama.model import LlamaTpPartModel
from slora.utils.infer_utils import nvtx_decorator


class Llama2TpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Llama2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Llama2TransformerLayerInfer

    @nvtx_decorator('Llama2TpPartModel __init__')
    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num, mem_adapter_size, load_way="HF", mode=[],
                 dummy=False):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size, load_way, mode, dummy=dummy)
    

    @nvtx_decorator('Llama2TpPartModel _init_config')
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return 
    
    @nvtx_decorator('Llama2TpPartModel _verify_params')
    def _verify_params(self):
        assert self.load_way == "HF", "llama only support HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    @nvtx_decorator('Llama2TpPartModel _init_mem_manager')
    def _init_mem_manager(self):
        # print("in llama2/model: self.mem_manager: head_num %d head_dim %d layer_num %d" %(self.config["num_attention_heads"] // self.world_size_,
        #                                                                            self.config["hidden_size"] // self.config["num_attention_heads"],
        #                                                                            self.config["num_hidden_layers"]))
        print('tot_size=self.max_total_token_num + self.mem_adapter_size: ',
              self.max_total_token_num + self.mem_adapter_size)
        self.mem_manager = self.memory_manager_class(tot_size=self.max_total_token_num + self.mem_adapter_size, 
                                                     cache_size=self.max_total_token_num,
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"])
        return
    
    @nvtx_decorator('Llama2TpPartModel _init_some_value')
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return
