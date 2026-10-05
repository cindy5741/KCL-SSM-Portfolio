import time
import torch
from utils_train import *
from tqdm import tqdm
from loss import CLoss, StudentLoss
import torch.nn as nn
import torch.nn.functional as F
from model_train import  CLBERT
import numpy as np

class Trainer():
    def __init__(self, args, tokenizer, device):
        super().__init__()
        self.args = args # initialize hyperparameters
        self.device = device # set device
        self.tokenizer = tokenizer
    
    # Train CL BERT (CL loss)
    def CL_train(self, model, train_loader, optimizer):
        
        # 
        #  
        #  NEW: Load Teacher Model for Knowledge Distillation
        #  
        teacher_model = None
        if self.args.get('use_teacher_kd', False): 
            print('\n==========  Loading Teacher Model  ==========\n')
            # 載入預訓練好的 CLBERT 模型作為 Teacher
            teacher_model_path = self.args["teacher_model_dir"] + self.args['teacher_ckpt']
            # 使用 CLBERT 類別來載入模型 (CLBERT 來自 model_train.py)
            teacher_model = CLBERT.from_pretrained(teacher_model_path, args=self.args).to(self.device)
            teacher_model.eval() # 設定為評估模式，固定 Teacher 模型的權重
            print(f'Teacher model loaded from: {teacher_model_path}')
        
        # === NEW: 初始化 student：拷貝 teacher 前 N 層 ===
        if (teacher_model is not None) and self.args.get("init_student_from_teacher", False):
            n_layers = self.args.get("student_num_layers", 10)
            self._init_student_from_teacher(model, teacher_model, n_layers)

        # Build teacher standard-question embedding bank (e.g., 250 std questions)
        teacher_bank = None
        if teacher_model is not None:
            teacher_bank = self._build_teacher_bank(train_loader, teacher_model)

        # 
        #  
        #  
        
        # define the save path
        save_model_path = self.args["model_dir"] + self.args["model_name"] + '_' + self.args["time"].split('_')[0]
        
        # define what loss function we use in CL
        criterion = StudentLoss(temperature=self.args['temperature'],
                                   alpha=self.args.get('alpha', 1.0),
                                   beta=self.args.get('beta', 1.0),
                                   lam=self.args.get('lambda', 0.5))
        best_loss = 10e9 # the record of loss to determine when to save model
        
        # for loop of training epochs
        for epoch in range(self.args["epochs"]):
            ## Training Phase ##
            step_count = 0 # the amount of batched dataloader
            train_loss = 0.0 # the record of training loss

            loop = tqdm((train_loader),total=len(train_loader), leave=False) # build tqdm to evaluate time consuming
            
            # Start Training
            model.train() # set model (Student) to train mode

            for data, labels in loop: # get data from dataloader
        
                # 
                #  Student Features (f1, f2)
                # 
                ids = data['input_ids'] # D = [bsz, 2, text_len] 
                input_ids = torch.cat([ids[:,0], ids[:,1]], dim=0).to(self.device) # D = [bsz*2, text_len]
                mask = data['attention_mask'] # D = [bsz, 2, text_len]
                attention_mask = torch.cat([mask[:,0], mask[:,1]], dim=0).to(self.device) # D = [bsz*2, text_len]
                types = data['token_type_ids'] # D = [bsz, 2, text_len]
                token_type_ids = torch.cat([types[:,0], types[:,1]], dim=0).to(self.device) # D = [bsz*2, text_len]
                labels = labels.to(self.device) # D = [bsz]
                bsz = labels.shape[0] # record the batch size of data
                
                # 
                #  
                #  NEW: Get Teacher Features (f_teacher)
                #  
                teacher_features = None
                if teacher_model is not None:
                    # 從 data 字典中獲取 'std_' 開頭的 features (來自 dataset.py)
                    std_ids = data['std_input_ids'].to(self.device) # D = [bsz, text_len]
                    std_mask = data['std_attention_mask'].to(self.device)
                    std_types = data['std_token_type_ids'].to(self.device)
                    
                    if std_ids.dim() == 3:
                        std_ids = std_ids.squeeze(1)
                        std_mask = std_mask.squeeze(1)
                        std_types = std_types.squeeze(1)
                    
                    with torch.no_grad(): # Teacher is fixed, no gradient
                        teacher_features = teacher_model(input_ids = std_ids,
                                                    attention_mask = std_mask,
                                                    token_type_ids = std_types)
                        # 正規化 Teacher features
                        teacher_features = F.normalize(teacher_features,dim=1) 
                # 
                #  
                #  
                
                # initialize the gradient
                optimizer.zero_grad()
                
                # get representation from model (the student encoder)
                # D = [bsz*2, emb_dim]
                features = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids)
                features = F.normalize(features,dim=1)
                # get the two features corresponding to two augmented texts
                f1, f2 = torch.split(features, [bsz, bsz], dim=0) # D_f1 = [bsz, emb_dim], D_f2 = [bsz, emb_dim]
                logits = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # D = [bsz, 2, emb_dim]
                
                # 
                #  
                #  
                #  NEW: calculate the Contrastive Loss (Student CL + KD)
                # Eq.(6)(7)(8) student loss
                # Query embedding z_i^s: average over two augmented views
                z_query_s = logits.mean(dim=1)  # [bsz, dim]

                # Student embedding of standard question z_ci^s (current student)
                std_ids_s = data['std_input_ids'].to(self.device)
                std_mask_s = data['std_attention_mask'].to(self.device)
                std_types_s = data['std_token_type_ids'].to(self.device)
                if std_ids_s.dim() == 3:
                    std_ids_s = std_ids_s.squeeze(1)
                    std_mask_s = std_mask_s.squeeze(1)
                    std_types_s = std_types_s.squeeze(1)
                z_ci_s = model(input_ids=std_ids_s, attention_mask=std_mask_s, token_type_ids=std_types_s)
                z_ci_s = F.normalize(z_ci_s, dim=1)

                if teacher_features is None or teacher_bank is None:
                    raise RuntimeError('StudentLoss requires teacher_features and teacher_bank, but got None. '
                                       'Please set args["use_teacher_kd"]=True and provide teacher checkpoints.')

                # Eq.(7): z_ci mix
                z_ci = criterion.mix_z_ci(z_ci_s, teacher_features)

                # Eq.(6)(8): final loss
                loss = criterion(z_query_s, labels, z_ci=z_ci, teacher_bank=teacher_bank)
                # 
                #  
                #  
                
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
            
            # 
            #  (底下儲存模型的程式碼保持不變)
            # 
            if self.args['save_best']:
                if train_loss <= best_loss:
                    model.save_pretrained(save_model_path)
                    self.tokenizer.save_pretrained(save_model_path)
                    # print(f'Model saved to ==> {save_model_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                    print(f'Model saved to ==> {save_model_path}')
                    best_loss = train_loss # update record
        
        if not self.args['save_best']:
            model.save_pretrained(save_model_path)
            self.tokenizer.save_pretrained(save_model_path)
            # print(f'Model saved to ==> {save_model_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            print(f'Model saved to ==> {save_model_path}')

        # record the result and save it
        res_fname = f"{save_model_path.split('/')[-1]}"
        content = f'==== Hyperparameters ====\n{self.args}\n==== Results ====\n\tloss\ntrain | {train_loss:.4f}'
        # save_result(res_fname, content) # save parameters and metrics result
    
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
    @torch.no_grad()
    def _encode_texts(self, model, texts, max_length=64, batch_size=64):
        """
        用 teacher_model 把 texts encode 成 normalized embeddings.
        這裡假設 teacher_model(...) 回傳 embedding tensor [bsz, dim]
        若你 teacher_model 回傳 tuple/dict，也有做相容處理。
        """
        model.eval()
        all_emb = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                token_type_ids=enc.get("token_type_ids", None),
            )

            # --- 相容不同輸出型態 ---
            if torch.is_tensor(out):
                emb = out
            elif isinstance(out, (tuple, list)):
                emb = out[-1]
            elif isinstance(out, dict):
                if "emb" in out:
                    emb = out["emb"]
                elif "pooler_output" in out:
                    emb = out["pooler_output"]
                elif "sentence_embedding" in out:
                    emb = out["sentence_embedding"]
                else:
                    raise ValueError(f"Teacher output dict keys not recognized: {out.keys()}")
            else:
                raise ValueError(f"Unsupported teacher output type: {type(out)}")
            # ----------------------

            emb = F.normalize(emb, dim=1)
            all_emb.append(emb.detach().cpu())

        return torch.cat(all_emb, dim=0)  # [N, dim]

    @torch.no_grad()
    def _build_teacher_bank(self, train_loader, teacher_model):
        """
        依公式(6) 的 teacher bank：{z_k^t}
        這裡用「每個 LABEL 取一條代表性 QUESTION」建立 bank。
        """
        teacher_model.eval()

        # 1) 從 dataset 拿 df
        if not hasattr(train_loader.dataset, "df"):
            raise AttributeError(
                "train_loader.dataset 沒有 df 屬性，無法建立 teacher bank。"
                "請確認 dataset.py 的 Dataset 是否有保留 self.df"
            )
        df = train_loader.dataset.df

        # 2) 自動找 label 欄位
        if "LABEL" in df.columns:
            label_col = "LABEL"
        elif "label" in df.columns:
            label_col = "label"
        else:
            raise KeyError(f"df 找不到 LABEL 欄位，現有 columns: {list(df.columns)}")

        # 3) 自動找 question/text 欄位（依你資料可能叫不同名字）
        possible_q_cols = ["QUESTION", "question", "text", "sentence", "utterance", "query"]
        q_col = None
        for c in possible_q_cols:
            if c in df.columns:
                q_col = c
                break
        if q_col is None:
            raise KeyError(f"df 找不到文字欄位(QUESTION/text/...)，現有 columns: {list(df.columns)}")

        # 4) 每個 label 取第一筆代表句（你也可以改成取 std_question 欄位）
        rep = df[[label_col, q_col]].dropna().drop_duplicates(subset=[label_col], keep="first")
        rep = rep.sort_values(by=label_col).reset_index(drop=True)

        texts = rep[q_col].astype(str).tolist()
        if len(texts) == 0:
            raise RuntimeError("teacher bank texts 為空，請檢查 df 是否有資料或欄位是否正確。")

        # 5) encode 成 bank embeddings
        bank = self._encode_texts(
            model=teacher_model,
            texts=texts,
            max_length=getattr(self.args, "max_length", 64) if hasattr(self.args, "max_length") else 64,
            batch_size=self.args.get("bank_batch_size", 64) if isinstance(self.args, dict) else 64
        )

        # 回傳 CPU bank；你 loss 裡面可以搬到 GPU（或在 loss 裡自動搬）
        return bank  # [K, dim]

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
        # if test_loader: # if containing test loader
        #    if test_loader:
        #     print("[Info] test_loader provided but CLCEBERT path removed (not used). Skipping test evaluation.")

        #     # test_loss, test_acc, test_f1, test_rec, test_prec = self.evaluate_dataloader(test_loader, eval_model, loss_fct, automodel)
            
        #     print('         loss     acc     f1      rec    prec')
        #     print('test   | %.4f, %.4f, %.4f, %.4f, %.4f'%(test_loss, test_acc, test_f1, test_rec, test_prec))

        #     if not self.args['save_best']:
        #         content = ('==== Hyperparameters ====\n'
        #         f'{self.args}\n'
        #         '==== Results ====\n'
        #         '\tLoss   Accuracy F1score Recall Precision\n'
        #         f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
        #         f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}\n'
        #         f'test  | {test_loss:.4f}\t{test_acc:.4f}\t{test_f1:.4f}\t{test_rec:.4f}\t{test_prec:.4f}')
        #     else:
        #         content = ('==== Hyperparameters ====\n'
        #         f'{self.args}\n'
        #         '==== Results ====\n')+res+(f'test  | {test_loss:.4f}\t{test_acc:.4f}\t{test_f1:.4f}\t{test_rec:.4f}\t{test_prec:.4f}')
        # else:
        #     if not self.args['save_best']:
        #         content = ('==== Hyperparameters ====\n'
        #         f'{self.args}\n'
        #         '==== Results ====\n'
        #         '\tLoss   Accuracy F1score Recall Precision\n'
        #         f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
        #         f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}')
        #     else:
        #         content = ('==== Hyperparameters ====\n'
        #         f'{self.args}\n'
        #         '==== Results ====\n')+res
        # res_fname = f"{save_model_path.split('/')[-1]}"
        # save_result(res_fname, content) # save parameters and metrics result
                # record the result and save it
        if not self.args['save_best']:
            content = ('==== Hyperparameters ====\n'
                       f'{self.args}\n'
                       '==== Results ====\n'
                       '\tLoss   Accuracy F1score Recall Precision\n'
                       f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
                       f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}')
        else:
            # save_best=True 時，這份程式原本就是用 res 來寫檔
            # 但 res 可能不存在，所以用保底版本
            content = ('==== Hyperparameters ====\n'
                       f'{self.args}\n'
                       '==== Results ====\n'
                       '\tLoss   Accuracy F1score Recall Precision\n'
                       f'train | {train_loss:.4f}\t{train_acc:.4f}\t{train_f1:.4f}\t{train_rec:.4f}\t{train_prec:.4f}\n'
                       f'val   | {val_loss:.4f}\t{val_acc:.4f}\t{val_f1:.4f}\t{val_rec:.4f}\t{val_prec:.4f}')

        return  content
    
    def _init_student_from_teacher(self, student_model, teacher_model, n_layers: int):
        """
        把 teacher 的 embeddings + 前 n_layers encoder 權重拷貝到 student。
        讓 student 不要從亂數開始學。
        """
        s = student_model.encoder
        t = teacher_model.encoder

        # 1) embeddings
        s.embeddings.load_state_dict(t.embeddings.state_dict())

        # 2) encoder layers 0..n_layers-1
        max_copy = min(n_layers, len(s.encoder.layer), len(t.encoder.layer))
        for i in range(max_copy):
            s.encoder.layer[i].load_state_dict(t.encoder.layer[i].state_dict())


        # 3) pooler（如果有）
        if hasattr(s, "pooler") and hasattr(t, "pooler") and s.pooler is not None and t.pooler is not None:
            s.pooler.load_state_dict(t.pooler.state_dict())

        print(f"[Init] Copied teacher -> student for {n_layers} layers.")
        print("Student layers:", len(s.encoder.layer), "Teacher layers:", len(t.encoder.layer))



