import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig, AutoModelForSequenceClassification
import torch.nn.functional as F

# Different models
def build_model(args):

    print('\n==========  Build Model  ==========\n')
    model_type = args['model_type']
    assert model_type in ['CLBERT', 'CLCEBERT', 'CEBERT', 'FET-roberta']
    config = AutoConfig.from_pretrained(args['config'])
    if model_type == 'CLBERT':
        return CLBERT(config, args)
    elif model_type == 'CLCEBERT':
        return CLCEBERT(config, args)
    elif model_type == 'CEBERT':
        return CEBERT(config, args)
    elif model_type == 'FET-roberta':
        return AutoModelForSequenceClassification.from_pretrained(args['config'], num_labels=args['num_class'])

# CLBERT Model (use BERT as an encoder to generate features)
class CLBERT(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.encoder = AutoModel.from_pretrained(args['config'])

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):

        # get features from encoder
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # get contextual representation
        pooled_output = outputs.pooler_output

        # Normalize the output
        # return F.normalize(pooled_output,dim=1)
        return pooled_output
    
# NN (transform the dimension)
class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, args):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(args['emb_dim'], args['num_class']) # build a linear layer with input/ouput dimension

    def forward(self, features):
        return self.fc(features) # get transformed features

# CLBERT (use CLBERT as the encoder (fine-tuning) and complete classification task)
class CLCEBERT(PreTrainedModel):
    """CL encoder with classifier"""
    config_class = AutoConfig
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        # load pretrained encoder
        model = CLBERT.from_pretrained(self.args["model_dir"]+self.args['ckpt'], args)
        self.encoder = model

        # build linear classifier layer (if layers are more than 1, you need to add activation layer)
        self.dropout = nn.Dropout(args["dropout"]) # dropout layer is used to decline the impact of overfitting
        self.classifier = nn.Linear(config.hidden_size, self.args['num_class'])

    # the process of data in model
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        
        # get representation from encoder
        pooled_output = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        return self.classifier(self.dropout(pooled_output)) # get the corresponding dimension features for classification

# CEBERT model (use BERT as encoder to complete classification task)
class CEBERT(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.num_labels = args['num_class']
        self.bert = AutoModel.from_pretrained(args['config'])
        self.dropout = nn.Dropout(args["dropout"])
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids= None, attention_mask = None, token_type_ids= None, position_ids= None, 
                head_mask = None, inputs_embeds = None, labels = None, output_attentions = None, 
                output_hidden_states = None, return_dict = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
