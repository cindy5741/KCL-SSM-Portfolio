import os
import time
import torch
import pprint
from transformers import logging
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import ClassificationDataset
from preprocess_generate.preprocess import read_data
from hyperparameter import PARAM
from model_train import build_model
from trainer import Trainer
import warnings

if __name__ == '__main__':
    logging.set_verbosity_error() # disable warning about transformer
    warnings.filterwarnings('ignore') # cancel the warning message
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set our device
    tokenizer = AutoTokenizer.from_pretrained(PARAM["config"])
    print('\n==========  Load DataLoader  ==========')
    
    # Load data
    if not PARAM['has_val']: # if there is no validation data, view test data as validation set
        train_df, test_df = read_data(PARAM["data_dir"], has_val=False) # get DataFrame
        val_dataset = ClassificationDataset(test_df, PARAM, tokenizer, label_col=PARAM["label_col"], augmented=False) # transform to Dataset
        val_loader = DataLoader(val_dataset, batch_size=PARAM["batch_size"], shuffle = False) # use Dataloader to batched
        test_loader = None
        print('# of test data', len(val_dataset))
    else:
        train_df, val_df, test_df = read_data(PARAM["data_dir"]) # get DataFrame
        val_dataset = ClassificationDataset(val_df, PARAM, tokenizer, label_col=PARAM["label_col"], augmented=False) # transform to Dataset
        val_loader = DataLoader(val_dataset, batch_size=PARAM["batch_size"], shuffle = True) # use Dataloader to batched
        test_dataset = ClassificationDataset(test_df, PARAM, tokenizer, label_col=PARAM["label_col"], augmented=False) # transform to Dataset
        test_loader = DataLoader(test_dataset, batch_size=PARAM["batch_size"], shuffle = False) # use Dataloader to batched
        print('# of val data', len(val_dataset))
        print('# of test data', len(test_dataset))

    # According to our training objective to make our dataset
    if PARAM['train_objective'] == 'CL':
        train_dataset = ClassificationDataset(train_df, PARAM, tokenizer, label_col=PARAM["label_col"], augmented=True) # transform to Dataset, only augmented texts
    else:
        train_dataset = ClassificationDataset(train_df, PARAM, tokenizer, label_col=PARAM["label_col"], augmented=False) # transform to Dataset, only origin texts
        
    print('# of train data', len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=PARAM["batch_size"], shuffle = True) # use Dataloader to batched

    print('Completed Loading Data!\n')

    print('=======  Hyperparameters  =======\n')
    pprint.pprint(PARAM) # print all hyperparameters on screen
    st_time = time.time() # initialize the timer
    trainer = Trainer(PARAM, tokenizer, device) # build trainer

    # build model then training
    if PARAM['train_objective'] == 'CL':
        model = build_model(PARAM).to(device) # build CLBERT model
        print(f'{model}\n\n==========  Start Training  ==========\n') # print model structure
        optimizer = torch.optim.Adam(model.parameters(), lr=PARAM["lr"], betas=(0.9, 0.98), eps=1e-9) # define optimizer of model
        
        title, content = trainer.CL_train(model, train_loader, optimizer) # train "model" with "CL" loss

    elif PARAM['train_objective'] == 'CLCE' and PARAM['ckpt'] is not None: 
        model = build_model(PARAM).to(device) # build CLCEBERT model
        print(f'{model}\n\n==========  Start Training  ==========\n') # print model structure
        optimizer = torch.optim.Adam(model.parameters(), lr=PARAM["lr"], betas=(0.9, 0.98), eps=1e-9) # define optimizer of model
        title, content = trainer.train_dataloader(model, optimizer, train_loader, val_loader, test_loader, automodel=False) # train "full model" with "CE" loss

    else:
        raise NotImplementedError("Only CL or CLCE is supported.")

    print('==========  Completed Training!  ==========\ncost time: %.4f s'%(time.time() - st_time))
