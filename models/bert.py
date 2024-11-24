from torch import nn
from transformers import RobertaModel


class BERTEncoder(nn.Module):
    def __init__(self, model_name):
        super(BERTEncoder, self).__init__()
        self.bert = RobertaModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output