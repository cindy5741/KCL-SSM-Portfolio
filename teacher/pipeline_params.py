import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# === model and data settings ===
MODEL_DIR = './model/'
CL_model_path = 'CL_2025-10-07'
CLCE_model_path = 'MORE_2025-10-08'  
Q2L_FILE = './data/question_to_label.tsv'  

params = { 
        'config': 'hfl/chinese-roberta-wwm-ext', # 'bert-base-chinese', 'hfl/chinese-roberta-wwm-ext'
        'model_dir': MODEL_DIR,
        'batch_size': 256,
        'max_len': 30,
}

param_CLCE_MORE = params.copy()
param_CLCE_MORE.update({
    'dropout': 0.3, # how random amount will be give up
    'num_class': 252, # class of data / classification; Remember change data folder according to num_class
    'ckpt': CL_model_path, # load model checkpoint path
    'q2l_fname': Q2L_FILE,
})


