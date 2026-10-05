import time
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from pipeline_params import *
from model import CLCEBERT
from torch.utils.data import Dataset, DataLoader
from utils import LabelTransform, predict_single, predict_dataloader
from transformers import logging, AutoTokenizer

# build dataset
class PredictDataset(Dataset):
    def __init__(self, df, args, tokenizer, label_col='LABEL', augmented=False):
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args["max_len"]
        self.label_col = label_col
    
    def __len__(self):
        return len(self.df)

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
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids, 'input_text':input_text}
    
    # get single data
    def __getitem__(self, index):

        col_name = self.df.columns
        sentence = str(self.df[col_name[0]][index])

        label = None
        if self.label_col is not None:
            label = torch.tensor(int(self.df[self.label_col][index]), dtype=torch.long)

        inputs = self.tokenize(sentence)

        if label == None:
            return inputs
        else:
            return index, inputs, label



if __name__ == '__main__':
    
    ## General Settings ##
    logging.set_verbosity_error() # disable warning about transformer
    warnings.filterwarnings('ignore') # cancel the warning message
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set our device

    ## Get Transform Label Tool ##
    tranform_tool = LabelTransform('./data/new_all_data_label.tsv')

    st_time = time.time()
    print('---- Loading model (CLCE)')
    
    CLCE_model = CLCEBERT.from_pretrained(MODEL_DIR+CLCE_model_path, args=param_CLCE_MORE).to(device)

    # Route I
    #MASK_model, CE_model = get_route_I_model(CE_model_path, MASK_model_path, MASKCE_model_path, device, type=ROUTE_I, backbone=BACKBONE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR+CLCE_model_path)
    print('cost time: %.4f s  Completed'%(time.time()-st_time))
    
    print('---- Executing Prediction ----')

    # if you need result.tsv, please run above code:
    # test set evaluation ##
    # test_df = pd.read_csv('./data/test.tsv', sep='\t', usecols=['SAMPLE', 'LABEL']).dropna().reset_index(drop=True)
    # test_dataset = PredictDataset(test_df, params, tokenizer) # transform to Dataset
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False) # use Dataloader to batched
    # print('# of data: %d'%(len(test_loader)))
    
    # result = predict_dataloader(test_loader, tokenizer, None, None, None, CLCE_model, params, tranform_tool, save_prob=True)
    # result.to_csv('./result.tsv', sep='\t', index=False)
    
    st_time = time.time()
    ## single prediction ##
    text = '欸那個，我想請問我有分期罰單漏繳，請問這會有影響嗎?'
    top_p, top_label, pred_label, pred_class = predict_single(text, tokenizer, None, None, None, CLCE_model, params, tranform_tool, topK=1)
    print(text, top_p, top_label, pred_label, pred_class)

    print('cost time: %.4f s  Completed'%(time.time()-st_time))
