from datetime import datetime

PARAM = {
            # Task Setting
            'data_dir': '../data/', # data directory (load path)
            'model_dir': '../model/', # save model directory
            'model_name': 'CL_teacher', # 'CL', 'MORE'
            'model_type': 'CLBERT', # 'CLBERT', 'CLCEBERT'

            # General SettingS
            'has_val': True, # if there are validation datasets (True means including train/val/test)
            'time': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), # annotate start training time
			'seed': 1111, # random seed and model shuffle
            'save_best': True, # if True, model will be saved according to acc(CE) / loss (CL); else model will be saved until last epoch of training

            # BERT Setting
            'config': 'hfl/chinese-roberta-wwm-ext', # BERT pre-trained config. 'hfl/chinese-roberta-wwm-ext'
			'max_len': 30, # the max length of input tokens

            # Specified Model Setting
            'train_objective': 'CL', # 'CL', 'CLCE' means train CL + CE, 'CE': means train PLM + classifier
            'label_col': 'LABEL', # 'LABEL'

            # Hyperparameters
            'optimizer': 'Adam', # 'Adam', 
			'lr': 1e-5, # the learning rate 1.5E-5
			'epochs': 400, # training epochs
			'batch_size': 256, # batch size, depend on your GPU
			'dropout': 0.3, # how random amount will be give up
			'num_class': 107024, # class of data / classification; Remember change data folder according to num_class
            #If 'model_type'='CLBERT', 'num_class'=Number of training samples 107024
            'temperature': 0.1, # contrastive loss scale controll
        }
