import torch
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

# build dataset
class ClassificationDataset(Dataset):
    def __init__(self, df, args, tokenizer, label_col='LABEL', augmented=False):
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args["max_len"]
        self.num_class = args["num_class"]
        self.label_col = label_col
        self.augmented = augmented
        self.mlm_probability = 0.15
    
    def __len__(self):
        return len(self.df)

    # masking augmentation
    def mask_aug(self, inputs, special_tokens_mask = None):
        """
        From DiffuCSE train.py mask_tokens() l512-544
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        ---> Modified to only replacing to [MASK]
        
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # sample 15% to replaced
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs
        
    # transform text to its number
    def tokenize_to_tensor(self, input_text):
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length = self.max_len,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )
        return inputs

    # transform text to its number
    def tokenize(self, input_text):
        # print(input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length = self.max_len,
            truncation = True,
            padding = 'max_length'
        )
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids}
    
    # get single data
    def __getitem__(self, index):

        col_name = self.df.columns
        sentence1 = str(self.df[col_name[1]][index])
        
        label = None
        if self.label_col is not None:
            label = torch.tensor(int(self.df[self.label_col][index]), dtype=torch.long)

        if self.augmented:
            inputs_s1 = self.tokenize_to_tensor(sentence1)
            inputs_s2 = self.tokenize_to_tensor(sentence1)
            inputs_s1['input_ids'] = self.mask_aug(inputs_s1["input_ids"])
            inputs_s2['input_ids'] = self.mask_aug(inputs_s2["input_ids"])
        
            inputs = {}
            for key in inputs_s1:
                inputs[key] = torch.cat([inputs_s1[key], inputs_s2[key]])
        else:
            inputs = self.tokenize(sentence1)

        if label == None:
            return inputs
        elif self.args['model_name'] == 'MASKLESS':
            return (index, inputs), label
        else:
            return inputs, label
