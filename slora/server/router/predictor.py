import transformers
import torch
import numpy as np


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
MAX_OUTPUT_LENGTH = 1000
BUCKET_NUM = 10
PER_BUCKET_LENGTH = MAX_OUTPUT_LENGTH/BUCKET_NUM


class Predictor:
    def __init__(self, model_dir):
        self.model = self.load_predictor(model_dir)

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
        return model

    def predict(self, input_ids):
        i = torch.tensor(input_ids)
        output_ids = self.model(i.unsqueeze(0)).logits
        # print(output_ids.squeeze())
        return np.argmax(output_ids.detach().numpy())