import json
import numpy as np

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
MAX_OUTPUT_LENGTH = 1000
BUCKET_NUM = 10
PER_BUCKET_LENGTH = MAX_OUTPUT_LENGTH/BUCKET_NUM

RESERVED=[]

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### From: {from}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### From: {from}\n\n### Response:"
    ),
}

ADAPTER_DIRS = [
    '/workspace/S-LoRA/LLM-models/LLaMA-2-7b/Adapters/tloen/alpaca-lora-7b', 
    '/workspace/S-LoRA/LLM-models/LLaMA-2-7b/Adapters/MBZUAI/bactrian-x-llama-7b-lora',
    ]

def main():
    data_path = '/workspace/distill-bert/lora-inference/my_datasets/mixed_dataset.json'
    generate_data_path = './my_traces.json'
    
    with open(data_path, 'r') as f:
        datas = json.load(f)
        
    print(len(datas))
    
    prompt_input , prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
    
    sources_inputs = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in datas
        ]
    
    sources_outputs = [
        example['output'] for example in datas
    ]
    
    adapters = [
        ADAPTER_DIRS[0] if ADAPTER_DIRS[0].split('/')[-1] in example['from'] else ADAPTER_DIRS[1] for example in datas 
    ]
    
    # print(sources_inputs[0:5])
    # print(sources_outputs[0:5])
    
    conversations = []
    
    for idx, (in_str, out_str) in enumerate(zip(sources_inputs, sources_outputs)):
        input_con = {'content': in_str, 'tstamp':np.random.uniform(0,10)}
        output_con = {'content': out_str, 'tstamp':np.random.uniform(10,20)}
        con = [input_con, output_con]
        conversations.append(
            {
                "conversation": con,
                'model': adapters[idx],
                'tstamp': np.random.uniform(input_con['tstamp'], output_con['tstamp']),
            }
        )
    
    with open(generate_data_path, 'w') as f:
        json.dump(conversations, f)
        
        
    
        
    

if __name__ == '__main__':
    main()