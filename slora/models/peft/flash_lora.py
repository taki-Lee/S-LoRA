import triton
import triton.language as tl

@triton.jit
def flash_mm_lora_mm_kernel(input_embs, base_layer_infer, base_layer_weight, 
                            y, x, w,
                            start_indicies,
                            lora_ranks,
                            loc_indicies,
                            indicies,
                            qkvo,
                            lora_scales):
    """
    ** for mm
        input_embs
        base_layer_infer
        base_layer_weight
    **

    ** for bgmv
        y: delta, cache
        x: reshaped input_embs
        w: key_buffer[layer_id]
        start_indicies: infer_adapter.a_start
        lora_ranks: infer_adapter.a_len
        loc_indicies: infer_adapter.a_loc
        indicies: self.req_bins
        qkvo: 0 for q, 1 for k, 2 for v, 3 for o
        lora_scales: infer_adapter.a_scaling
    **
    """
    ## q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
    

    ## dispatch_bgmv shrink


    ## dispatch_bgmv expand


    pass

@triton.jit
def triton_matmul():
    pass

def matmul_bgmv(input_embs, base_layer_infer, base_layer_weight, 
                y, x, w,
                start_indicies,
                lora_ranks,
                loc_indicies,
                indicies,
                qkvo,
                lora_scales):
    # matmul

    # bgmv


    pass

