import time
import torch
from utils_train import *
from tqdm import tqdm
from loss import CLoss
import torch.nn as nn
import torch.nn.functional as F
from model_train import CLCEBERT
from transformers import AutoModelForSequenceClassification

class Trainer():
    def __init__(self, args, tokenizer, device):
        super().__init__()
        self.args = args # initialize hyperparameters
        self.device = device # set device
        self.tokenizer = tokenizer
    
    # Train CL BERT (CL loss)
    def CL_train(self, model, train_loader, optimizer):
        
        # define the save path
        save_model_path = self.args["model_dir"] + self.args["model_name"] + '_' + self.args["time"].split('_')[0]
        
        # define what loss function we use in CL
        criterion = CLoss(temperature=self.args['temperature'])
        best_loss = 10e9 # the record of loss to determine when to save model
        
        # for loop of training epochs
        for epoch in range(self.args["epochs"]):
            ## Training Phase ##
            step_count = 0 # the amount of batched dataloader
            train_loss = 0.0 # the record of training loss

            loop = tqdm((train_loader),total=len(train_loader), leave=False) # build tqdm to evaluate time consuming
            
            # Start Training
            model.train() # set model to train mode

            for data, labels in loop: # get data from dataloader
        
                # in CL, we will concat the two augmented texts as input
                # put all data to the device that same as model
                ids = data['input_ids'] # D = [bsz, 2, text_len] 
                input_ids = torch.cat([ids[:,0], ids[:,1]], dim=0).to(self.device) # D = [bsz*2, text_len]
                mask = data['attention_mask'] # D = [bsz, 2, text_len]
                attention_mask = torch.cat([mask[:,0], mask[:,1]], dim=0).to(self.device) # D = [bsz*2, text_len]
                types = data['token_type_ids'] # D = [bsz, 2, text_len]
                token_type_ids = torch.cat([types[:,0], types[:,1]], dim=0).to(self.device) # D = [bsz*2, text_len]
                labels = labels.to(self.device) # D = [bsz]
                bsz = labels.shape[0] # record the batch size of data
                
                # initialize the gradient
                optimizer.zero_grad()
                
                # get representation from model (the encoder we defined)
                # D = [bsz*2, emb_dim]
                features = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids)
                features = F.normalize(features,dim=1)
                # get the two features corresponding to two augmented texts
                f1, f2 = torch.split(features, [bsz, bsz], dim=0) # D_f1 = [bsz, emb_dim], D_f2 = [bsz, emb_dim]
                logits = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # D = [bsz, 2, emb_dim]
                
                # calculate the Contrastive Loss
                loss = criterion(logits, labels)
                
                # loss backpropagation and update optimizer
                loss.backward()
                optimizer.step()
                
                # update record
                train_loss += loss.item()
                loop.set_postfix(train_loss=loss.item())
                step_count += 1

            # average the record of the total amount of batch dataloader in each epoch
            train_loss = train_loss / step_count
            
            print('\n[epoch %d] train_loss: %.4f'%(epoch+1, train_loss))
            
            # if set the parapmeter to True, we will save the model when it is better than the previous one
            if self.args['save_best']:
                if train_loss <= best_loss:
                    model.save_pretrained(save_model_path)
                    self.tokenizer.save_pretrained(save_model_path)
                    print(f'Model saved to ==> {save_model_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                    best_loss = train_loss # update record
        # if set the parameters to False, we only save the model until the last epoch
        if not self.args['save_best']:
            model.save_pretrained(save_model_path)
            self.tokenizer.save_pretrained(save_model_path)
            print(f'Model saved to ==> {save_model_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

        # record the result and save it
        res_fname = f"{save_model_path.split('/')[-1]}"
        content = f'==== Hyperparameters ====\n{self.args}\n==== Results ====\n\tloss\ntrain | {train_loss:.4f}'
        save_result(res_fname, content) # save parameters and metrics result
    
        return res_fname, content

    # evaluate BERT
    def evaluate_dataloader(self, data_loader, model, loss_fct, automodel=False):
        loop = tqdm((data_loader),total=len(data_loader), leave=False) # build tqdm to evaluate time consuming
        val_loss, val_acc, val_f1, val_rec, val_prec = 0.0, 0.0, 0.0, 0.0, 0.0 # initialize the record
        step_count = 0 # the amount of batched dataloader
        
        model.eval() # set model to eval mode
        with torch.no_grad(): # disable the gradient updated to fix the model parameters
            for tuple, labels in loop: # get data from dataloader
                
                data = tuple
                
                # in CE, we will use the origin texts as input
                # put all data to the device that same as model
                ids = data['input_ids'].to(self.device) # D = [bsz, text_len] 
                masks = data['attention_mask'].to(self.device) # D = [bsz, text_len] 
                token_type_ids = data['token_type_ids'].to(self.device) # D = [bsz, text_len] 
                labels = labels.to(self.device) # D = [bsz] 

                # get representation from model (the encoder we defined)
                # D = [bsz, num_class]
                if not automodel:
                    logits = model(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                else:
                    outputs = model(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                    logits = outputs.logits
                
                # calculate CrossEntropyLoss
                loss = loss_fct(logits, labels)
                # get evaluation metrics
                acc, f1, rec, prec = cal_metrics(logits, labels)

                # update record
                val_loss += loss.item()
                val_acc += acc
                val_f1 += f1
                val_rec += rec
                val_prec += prec
                step_count+=1
                loop.set_postfix(val_loss=loss.item(), val_acc=acc, val_f1=f1, val_rec=rec, val_prec=prec)
            # average the record of the total amount of batch dataloader in each epoch
            val_loss = val_loss / step_count
            val_acc = val_acc / step_count
            val_f1 = val_f1 / step_count
            val_rec = val_rec / step_count
            val_prec = val_prec / step_count
        return val_loss, val_acc, val_f1, val_rec, val_prec

    # Train BERT model
    def train_dataloader(self, model, optimizer, train_loader, val_loader, test_loader=None, automodel=False):
        save_model_path = self.args["model_dir"] + self.args["model_name"] + '_' + self.args["time"].split('_')[0] # define the save path
        best_acc = 0.0 # the record of acc to determine when to save model
        loss_fct = nn.CrossEntropyLoss() # define what loss function we use in CE

        # for loop of training epochs
        for epoch in range(self.args["epochs"]):
            loop = tqdm((train_loader),total=len(train_loader), leave=False) # build tqdm to evaluate time consuming
            st_time = time.time() # initialize the timer at the beginning of each epoch
            train_loss, train_acc, train_f1, train_rec, train_prec = 0.0, 0.0, 0.0, 0.0, 0.0 # initialize the record
            step_count = 0 # the amount of batched dataloader
    
            model.train() # set model to train mode
            for tuple, labels in loop: # get data from dataloader
                
                data = tuple
                
                # in CE, we will use the origin texts as input
                # put all data to the device that same as model
                ids = data['input_ids'].to(self.device) # D = [bsz, text_len] 
                masks = data['attention_mask'].to(self.device) # D = [bsz, text_len] 
                token_type_ids = data['token_type_ids'].to(self.device) # D = [bsz, text_len] 
                labels = labels.to(self.device) # D = [bsz] 
                
                optimizer.zero_grad() # initialize the gradient
                
                # get representation from model (the encoder we defined)
                # D = [bsz, num_class]
                if not automodel:
                    logits = model(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                else:
                    outputs = model(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                    logits = outputs.logits                

                # calculate CrossEntropyLoss
                loss = loss_fct(logits, labels)
                # get evaluation metrics
                acc, f1, rec, prec = cal_metrics(logits, labels)

                # loss backpropagation and update optimizer
                loss.backward()
                optimizer.step()

                # update record
                train_loss += loss.item()
                train_acc += acc
                train_f1 += f1
                train_rec += rec
                train_prec += prec
                step_count += 1
                loop.set_postfix(train_loss=loss.item(), train_acc=acc, train_f1=f1, train_rec=rec, train_prec=prec)

            # get evaluation performance of the model for the validation set
            val_loss, val_acc, val_f1, val_rec, val_prec = self.evaluate_dataloader(val_loader, model, loss_fct, automodel)
            # average the record of the total amount of batch dataloader in each epoch
            train_loss = train_loss / step_count
            train_acc = train_acc / step_count
            train_f1 = train_f1 / step_count
            train_rec = train_rec / step_count
            train_prec = train_prec / step_count

            # show the metrics record each epoch on screen
            print('\n[epoch %d] cost time: %.4f s'%(epoch + 1, time.time() - st_time))
            print('         loss     acc     f1      rec    prec')
            print('train | %.4f, %.4f, %.4f, %.4f, %.4f'%(train_loss, train_acc, train_f1, train_rec, train_prec))
            print('val   | %.4f, %.4f, %.4f, %.4f, %.4f\n'%(val_loss, val_acc, val_f1, val_rec, val_prec))

            # if set the parameters to True, we will save the model when it is better than the previous one
            if self.args['save_best']:
                if val_acc > best_acc:
                    model.save_pretrained(save_model_path)
                    self.tokenizer.save_pretrained(save_model_path)
                    print(f'Model saved to ==> {save_model_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                    best_acc = val_acc
                    res = ('\tLoss   Accuracy F1score Recall Precision\n'
                            f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
                            f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}\n')
        
        # if set the parameters to False, we only save the model until the last epoch
        if not self.args['save_best']:
            model.save_pretrained(save_model_path)
            self.tokenizer.save_pretrained(save_model_path)
            print(f'Model saved to ==> {save_model_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

        # record the result and save it
        if test_loader: # if containing test loader
            if self.args['model_type'] == 'CLCEBERT':
                eval_model = CLCEBERT.from_pretrained(save_model_path, args=self.args).to(self.device)
            else:
                raise NotImplementedError("Only CLCEBERT is supported.")

            test_loss, test_acc, test_f1, test_rec, test_prec = self.evaluate_dataloader(test_loader, eval_model, loss_fct, automodel)
            
            print('         loss     acc     f1      rec    prec')
            print('test   | %.4f, %.4f, %.4f, %.4f, %.4f'%(test_loss, test_acc, test_f1, test_rec, test_prec))

            if not self.args['save_best']:
                content = ('==== Hyperparameters ====\n'
                f'{self.args}\n'
                '==== Results ====\n'
                '\tLoss   Accuracy F1score Recall Precision\n'
                f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
                f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}\n'
                f'test  | {test_loss:.4f}\t{test_acc:.4f}\t{test_f1:.4f}\t{test_rec:.4f}\t{test_prec:.4f}')
            else:
                content = ('==== Hyperparameters ====\n'
                f'{self.args}\n'
                '==== Results ====\n')+res+(f'test  | {test_loss:.4f}\t{test_acc:.4f}\t{test_f1:.4f}\t{test_rec:.4f}\t{test_prec:.4f}')
        else:
            if not self.args['save_best']:
                content = ('==== Hyperparameters ====\n'
                f'{self.args}\n'
                '==== Results ====\n'
                '\tLoss   Accuracy F1score Recall Precision\n'
                f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
                f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}')
            else:
                content = ('==== Hyperparameters ====\n'
                f'{self.args}\n'
                '==== Results ====\n')+res
        res_fname = f"{save_model_path.split('/')[-1]}"
        save_result(res_fname, content) # save parameters and metrics result
    
        return res_fname, content
    
    


