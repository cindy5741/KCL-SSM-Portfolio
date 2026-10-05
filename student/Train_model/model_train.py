import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import PretrainedConfig

def build_model(args):
    print('\n==========  Build Model  ==========\n')

    config = AutoConfig.from_pretrained(args['config'])
    config.num_hidden_layers = args.get('student_num_layers', 10)

    return CLBERT(config, args)

# CLBERT Model (use BERT as an encoder to generate features)
class CLBERT(PreTrainedModel):
    config_class = AutoConfig  
    base_model_prefix = "encoder" 

    def __init__(self, config, args):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        print("[Model] num_hidden_layers =", self.encoder.config.num_hidden_layers)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        return outputs.pooler_output


