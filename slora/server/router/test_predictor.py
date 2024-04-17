from predictor import Predictor, PER_BUCKET_LENGTH, BUCKET_NUM, MAX_OUTPUT_LENGTH
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
model_path = "/workspace/distill-bert/knowledge-distillation-transformers-pytorch-sagemaker/lslee/new_dataset/distill-bert-extended-mixed-40/checkpoint-21200"
model = Predictor(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def load_dataset(path):
    with open(path, 'r') as f:
        datas = json.load(f)
    
    inputs = []
    outputs = []
    for data in datas:
        inputs.append(str(data['conversation'][0]['content']))
        outputs.append(str(data['conversation'][1]['content']))
    # print(inputs[:5])
    # print(outputs[:5])
    return inputs, outputs

def evaluate(inputs, outputs):
    correct = 0
    total = 0
    tru_buckets = [0] * BUCKET_NUM
    pre_buckets = [0] * BUCKET_NUM
    in_buckets = [0] * BUCKET_NUM
    for input_str, output_str in tqdm(zip(inputs, outputs)):
        input_ids = tokenizer(input_str).input_ids

        output_ids = tokenizer(output_str).input_ids
        pre_label = model.predict(input_ids, len(output_ids))
        predict = np.argmax(pre_label)

        tru_buckets[len(output_ids)//PER_BUCKET_LENGTH] += 1
        pre_buckets[predict] += 1
        in_buckets[len(input_ids)//PER_BUCKET_LENGTH] += 1

        if len(output_ids)//PER_BUCKET_LENGTH == predict:
            correct += 1
        total += 1
    
    print("accuracy: ", float(correct)/total)
    print(tru_buckets)
    print(pre_buckets)
    print(in_buckets)


inputs, outputs = load_dataset('/workspace/S-LoRA/benchmarks/real_trace/my_traces.json')
evaluate(inputs[:10000], outputs[:10000])