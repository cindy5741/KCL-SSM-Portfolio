import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
# from Train_model.model import CEBERT
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from Train_model.preprocess_generate.preprocess import read_data
#from Train_model.dataset import RelationDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


# softmax function (numpy version)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# evaluate dataloader
def predict_concat_relation(model, data_loader, device):
    total_logits = None
    model.eval()        
    with torch.no_grad():
        for data in data_loader:
            ids = data['input_ids'].to(device) # D = [bsz, text_len] 
            masks = data['attention_mask'].to(device) # D = [bsz, text_len] 
            token_type_ids = data['token_type_ids'].to(device) # D = [bsz, text_len] 
             
            # forward pass
            logits = model(input_ids = ids, 
                        token_type_ids = token_type_ids, 
                        attention_mask = masks)
            
            if total_logits is None:
                total_logits = logits
            else:
                total_logits = torch.cat([total_logits, logits], dim=0)
    
    total_logits = total_logits.detach().cpu().numpy()
    probs = np.array([softmax(logit) for logit in total_logits])[:, 1]
    
    return total_logits, probs

# get a standard question list
def get_std_qs(q2l_fname):
    df = pd.read_csv(q2l_fname, sep='\t')
    std_qs = df.QUESTION
    std2label, label2std = dict(), dict()
    for idx, q in enumerate(df['QUESTION']):
        std2label[q] = df['LABEL'][idx]
        label2std[df['LABEL'][idx]] = q
    return std_qs, std2label, label2std

# get top-K prediction from single query
def get_predtion(model, new_submit, tokenizer, params, device):

    df_rows = []
    std_qs, std2label, label2std = get_std_qs(params['q2l_fname'])
    res = []
    
    for i, std_q in enumerate(std_qs):
        df_rows.append({'SAMPLE': new_submit, 'CLASS':std_q.strip()})
    df = pd.DataFrame(df_rows, columns = ['SAMPLE', 'CLASS'])
    test_set = RelationDataset(df, params, tokenizer, label_col=None)
    test_loader = DataLoader(test_set, params['batch_size'], shuffle=False)

    test_logits, probs = predict_concat_relation(model, test_loader, device)

    for i, prob in enumerate(probs):
        res.append((i, std_qs[i].strip(), prob))

    res = sorted(res, key = lambda x: x[2], reverse = True)        
    
    topk_idx = [item[0] for item in res]
    topk_prob = [item[2] for item in res]
    
    res = np.zeros(len(std_qs))
    res[topk_idx] = topk_prob

    return torch.tensor(res)

# generate mask & save
def generate_mask_from_df(df, params, tokenizer, mask_model, device, save=True, save_name=None):
    topk_mask_prob = torch.empty((0, params['num_class']))

    loop = tqdm((df['SAMPLE']),total=len(df['SAMPLE']), leave=False)
    for sentence in loop:
        topk_mask_tmp = get_predtion(mask_model, sentence, tokenizer, params, device)
        topk_mask_prob = torch.cat((topk_mask_prob, topk_mask_tmp.unsqueeze(0)), dim=0)  # 疊加張量
    
    if save and save_name is not None:
        if not os.path.exists('./weight_mask/'):
            os.makedirs('./weight_mask/')
        torch.save(topk_mask_prob, './weight_mask/' + save_name)
    return topk_mask_prob

# update weight of mask
def weighted_mask(mask):
    weight_metrics = {1: 20, 4: 2, 5: 1.5, 200:1.3}
    if len(mask.shape)<2:
        weight_indices = {k:[] for k in weight_metrics.keys()}
        for k, w in sorted(weight_metrics.items(), key=lambda x:x[0], reverse=False):
            weight_indices[k] = torch.topk(mask, k).indices
        mask[:] = 1 # Others
        for k, w in sorted(weight_metrics.items(), key=lambda x:x[0], reverse=True):
            indices = weight_indices[k]
            mask[indices] = w
    else:
        for i, each_mask_prob in enumerate(mask):
            weight_indices = {k:[] for k in weight_metrics.keys()}
            for k, w in sorted(weight_metrics.items(), key=lambda x:x[0], reverse=False):
                weight_indices[k] = torch.topk(each_mask_prob, k).indices
            each_mask_prob[:] = 1 # Others
            for k, w in sorted(weight_metrics.items(), key=lambda x:x[0], reverse=True):
                indices = weight_indices[k]
                each_mask_prob[indices] = w
    return mask
    
# generate mask
def generate_mask_from_sentece(sentence, params, tokenizer, mask_model, device):
    topk_mask_prob = torch.empty((0, params['num_class']))
    topk_mask_prob = get_predtion(mask_model, sentence, tokenizer, params, device)
    
    return topk_mask_prob

# generate mask
def generate_mask_from_dataloader(bz_data, params, tokenizer, mask_model, device):
    topk_mask_prob = torch.empty((0, params['num_class']))
    for sentence in bz_data:
        mask_tmp = get_predtion(mask_model, sentence, tokenizer, params, device)
        topk_mask_prob = torch.cat((topk_mask_prob, mask_tmp.unsqueeze(0)), dim=0)
    
    return topk_mask_prob

# the tool for transform label
class LabelTransform():
    def __init__(self, corpus='./data/new_all_data_label.tsv'):
        super().__init__()
        self.label_class, self.ce_label_class, self.label_ce_class, self.clce_label_class, self.label_clce_class = self.get_transform_corpus(corpus)

    def get_transform_corpus(self, fname):

        ce_label_class, label_ce_class = {}, {}
        clce_label_class, label_clce_class = {}, {}

        corpus = pd.read_csv(fname, sep='\t')
        ce_data = corpus[corpus['ROUTE']==0].reset_index(drop=True)
        clce_data = corpus[corpus['ROUTE']==1].reset_index(drop=True)

        for i in range(len(ce_data)):
            if ce_data['LABEL'][i] not in ce_label_class:
                ce_label_class[ce_data['LABEL'][i]] = ce_data['ORI_LABEL'][i]
            if ce_data['ORI_LABEL'][i] not in label_ce_class:
                label_ce_class[ce_data['ORI_LABEL'][i]] = ce_data['LABEL'][i]
        for i in range(len(clce_data)):
            if clce_data['LABEL'][i] not in clce_label_class:
                clce_label_class[clce_data['LABEL'][i]] = clce_data['ORI_LABEL'][i]
            if clce_data['ORI_LABEL'][i] not in label_clce_class:
                label_clce_class[clce_data['ORI_LABEL'][i]] = clce_data['LABEL'][i]

        label_class = {}
        for i in range(len(corpus)):
            if corpus['ORI_LABEL'][i] not in label_class:
                label_class[corpus['ORI_LABEL'][i]] = corpus['CLASS'][i]
        
        return label_class, ce_label_class, label_ce_class, clce_label_class, label_clce_class

    # transform global to local 
    def transform_to_custom_label(self, label, clce=False):
        if torch.is_tensor(label):
            labels = label.tolist()[0]
        else:
            labels = [label]
        pred = []
        for l in labels:
            if clce:
                pred.append(self.label_clce_class[l])
            else:
                pred.append(self.label_ce_class[l])
        if len(pred) == 1:
            return pred[0]
        else:
            return pred

    # transform local to global 
    def transform_to_global_label(self, label, clce=False):
        if torch.is_tensor(label):
            labels = label.tolist()[0]
        else:
            labels = [label]
        pred = []
        for l in labels:
            if clce:
                pred.append(self.clce_label_class[l])
            else:
                pred.append(self.ce_label_class[l])
        if len(pred) == 1:
            return pred[0]
        else:
            return pred
        
    # transform label to class 
    def transform_label_to_class(self, labels):
        if isinstance(labels, np.int64):
            labels = [labels]
        else:
            pass
        pred = []
        for l in labels:
            if isinstance(l, torch.Tensor):
                l = l.item()
            pred.append(self.label_class[l])
        if len(pred) == 1:
            return pred[0]
        else:
            return pred
    

# predict single sentence (without calculating the accuracy)
def predict_single(text, tokenizer, router, mask_model, ce_model, clce_model, params, tranform_tool, topK=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # router、ce_model、clce_model 可能為 None，要先判斷
    if router is not None:
        router.eval()
    if ce_model is not None:
        ce_model.eval()
    if clce_model is not None:
        clce_model.eval()

    with torch.no_grad():
        data = tokenizer.encode_plus(
            text,
            max_length=30,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        ids = data['input_ids'].to(device)
        masks = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        # 先決定 route (如果沒有 router，直接固定走 Route II)
        if router is not None:
            if params['backbone'] == 'CEBERT':
                logits = router(input_ids=ids, attention_mask=masks, token_type_ids=token_type_ids)
            else:
                outputs = router(input_ids=ids, attention_mask=masks, token_type_ids=token_type_ids)
                logits = outputs.logits
            prob = torch.nn.functional.softmax(logits, dim=1)
            _, route = prob.topk(1, dim=1)
            route = route.item()
        else:
            route = 1  # 固定走 Route II

        # 根據 route 決定用哪個模型
        if route == 0:
            # Route I
            if ce_model is None:
                raise ValueError("ce_model is None but route==0 requires ce_model.")
            if params['backbone'] == 'CEBERT':
                outputs = ce_model(input_ids=ids, attention_mask=masks, token_type_ids=token_type_ids)
            else:
                outputs = ce_model(input_ids=ids, attention_mask=masks, token_type_ids=token_type_ids)
                outputs = outputs.logits
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        else:
            # Route II
            if clce_model is None:
                raise ValueError("clce_model is None but route==1 requires clce_model.")
            outputs = clce_model(input_ids=ids, attention_mask=masks, token_type_ids=token_type_ids)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        logits = logits.detach().cpu()
        prob = torch.nn.functional.softmax(logits, dim=1)
        top_p, top_label = prob.topk(topK, dim=1)

        # 轉換標籤
        pred_label = top_label
        pred_class = tranform_tool.transform_label_to_class(pred_label)
        
        top_label = top_label.item()
        pred_label = pred_label.item()
        return top_p, top_label, pred_label, pred_class

def predict_single_topk(text, tokenizer, model, params, topK=50):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        data = tokenizer.encode_plus(
            text,
            max_length=30,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        ids = data['input_ids'].to(device)
        masks = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        outputs = model(input_ids=ids, attention_mask=masks, token_type_ids=token_type_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.softmax(logits, dim=1)

        top_p, top_label = probs.topk(topK, dim=1)
        return top_p.squeeze(0).tolist(), top_label.squeeze(0).tolist()


# predict several sentence (with calculateing the accuracy)
def predict_dataloader(data_loader, tokenizer, router, mask_model, ce_model, clce_model, params, tranform_tool, save_prob=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc = 0.0
    result = []

    if router is not None:
        router.eval()
    if ce_model is not None:
        ce_model.eval()
    if clce_model is not None:
        clce_model.eval()
    if mask_model is not None:
        mask_model.eval()
    with torch.no_grad(): # disable the gradient updated to fix the model parameters
        for index, data, labels in data_loader: # get data from dataloader
            
            # in CE, we will use the origin texts as input
            # put all data to the device that same as model
            # then multiple to correspond weight to the outputs from the model
            ids = data['input_ids'].to(device) # D = [bsz, text_len] 
            masks = data['attention_mask'].to(device) # D = [bsz, text_len] 
            token_type_ids = data['token_type_ids'].to(device) # D = [bsz, text_len]
            input_text = data['input_text'] # list = [bsz]

            if router is not None:
            # get route
                if params['backbone'] == 'CEBERT':
                    logits = router(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                else:
                    outputs = router(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                    logits = outputs.logits
                
                prob = F.softmax(logits, dim=1)
                _, route = prob.topk(1, dim = 1)
            else:
                # 只有 Route II
                route = torch.ones(1, dtype=torch.int64).to(device)  # 固定走 Route II
           
           
            # get representation from the model (the encoder we defined)
            if not route.item():
                #print('Go to --> Route I.')
                ## Route I.
                if params['backbone'] == 'CEBERT':
                    logits = ce_model(input_ids = ids, 
                                attention_mask = masks, 
                                token_type_ids = token_type_ids)
                else:
                    outputs = router(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)
                    logits = outputs.logits
                if mask_model is not None and params is not None:
                    mask_prob = generate_mask_from_dataloader(input_text, params, tokenizer, mask_model, device)
                    mask_prob = weighted_mask(mask_prob)
                    logits = logits * mask_prob.cuda()
            else:
               # print('Go to --> Route II.')
                ## Route II.
                logits = clce_model(input_ids = ids, 
                            attention_mask = masks, 
                            token_type_ids = token_type_ids)

            # get predicted label
            probs = F.softmax(logits, dim=1) # get each class-probs
            labels = labels.to(device) # D = [bsz]

            # Get top-1 prediction
            top_p, top_label = probs.topk(1, dim = 1)
            # 換算 label
            pred_label = top_label
            

            pred_class = tranform_tool.transform_label_to_class(pred_label)
            
            true_class = tranform_tool.transform_label_to_class(labels)

            # Compute the Accuracy
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            acc += (torch.tensor(pred_label).to(device) == labels).sum().item()/len(labels)

            pred = top_label.item()
            pred_probs = probs[0][pred].item()

            result.append({'text': input_text[0], 'pred_label': pred_label, 'pred_class': pred_class, 'probs': '{:.4f}'.format(pred_probs), 'true_label': labels.item(), 'true_class':true_class})

        acc /= len(data_loader)
        print('Accuracy: %.4f'%(acc))
        if save_prob:
            res_df = pd.DataFrame(result, columns=['text', 'pred_label', 'pred_class', 'probs', 'true_label', 'true_class'])
        else:
            res_df = pd.DataFrame(result, columns=['text', 'pred_label', 'pred_class', 'true_label', 'true_class'])
        # print(res_df.head(5))
        return res_df