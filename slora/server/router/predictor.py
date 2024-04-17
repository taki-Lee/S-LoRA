import transformers
import torch
import numpy as np
from slora.utils.infer_utils import calculate_time


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
MAX_OUTPUT_LENGTH = 500
BUCKET_NUM = 10
PER_BUCKET_LENGTH = MAX_OUTPUT_LENGTH//BUCKET_NUM


class Predictor:
    def __init__(self, model_dir):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.load_predictor(model_dir)
        self.correct = 0
        self.wrong = 0
        self.truth_buckets = [0]*BUCKET_NUM
        self.predict_buckets = [0]*BUCKET_NUM

    def load_predictor(self, model_dir):
        num_labels = BUCKET_NUM
        labels = range(BUCKET_NUM)
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=None,
        )
        model.to(self.device)
        return model

    # @calculate_time(show=True, min_cost_ms=0.01)
    def predict(self, input_ids, output_len):
        with torch.no_grad():
            i = torch.tensor(input_ids, device=self.device)
            output_ids = self.model(i.unsqueeze(0)).logits
            pre_label = np.argmax(output_ids.detach().cpu().numpy())
            tru_label = output_len//PER_BUCKET_LENGTH
        self.truth_buckets[tru_label] += 1
        self.predict_buckets[pre_label] += 1
        if tru_label == pre_label:
            self.correct += 1
        else:
            self.wrong += 1
        # print(output_ids.squeeze())
        return pre_label
    
    def cal_accuracy(self):
        if self.correct + self.wrong == 0:
            return None
        # print('truth_buckets:', self.truth_buckets)
        # print('predi_buckets:',self.predict_buckets)
        return float(self.correct) / (self.correct + self.wrong)