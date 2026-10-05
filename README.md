# KCL-SSM
## Contrastive Learning Based Sentence Semantic Matching with Knowledge Graph Augmentation
A sentence semantic matching framework for domain-specific question matching, integrating contrastive learning, knowledge graph-enhanced candidate retrieval, and hybrid scoring.
<img width="2401" height="988" alt="image" src="https://github.com/user-attachments/assets/58e27f52-f101-4c60-8242-bfcbb03ec929" />

## Overview
KCL-SSM is a sentence semantic matching framework designed for domain-specific question matching. It addresses common challenges in practical question-answering systems, including fine-grained intent overlap, colloquial expressions, and limited domain-specific labeled data.

The framework integrates three key components:

* **Contrastive Learning** for learning discriminative sentence representations.
* **Knowledge Graph-Enhanced Candidate Retrieval** for incorporating domain-specific structural knowledge into candidate selection.
* **Hybrid Scoring** for combining semantic similarity with knowledge-based retrieval signals.

The framework was evaluated on multiple public datasets and a real-world industrial dataset. Experimental results demonstrate the effectiveness of combining semantic learning with structured knowledge for domain-specific question matching.


# Environment Setting
Suggested running environments: Linux Ubuntu 22.04.4, Python 3.9

1. Install the Conda package for your system. The installation of the package can be found at <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">here<a/>

2. Create the NLP Conda environment. This may take a while, depending on the network status


整體流程圖


細節
<img width="1550" height="725" alt="image" src="https://github.com/user-attachments/assets/dbcd0c3f-da4d-4350-8d83-6223aae426aa" />
<img width="1745" height="406" alt="image" src="https://github.com/user-attachments/assets/83f46ac0-0b67-4e88-9504-c4633a1a8e7b" />
<img width="2204" height="403" alt="image" src="https://github.com/user-attachments/assets/9278ce4c-b617-402c-8e4b-0ba4d44ea2c1" />
<img width="1575" height="444" alt="image" src="https://github.com/user-attachments/assets/d526cf14-1913-4306-afb4-ae0a0099d551" />


```
conda create -n NLP python=3.9
```

3. Activate your NLP Conda environment. 

```
conda activate NLP
```

4. If you want to leave the NLP Conda environment, please type:

```
conda deactivate NLP
```

5. If this is the first time you install, run the following command to install the necessary packages. 

```
pip install -r requirements.txt
```

# Teacher model
Switch to the `teacher` directory.
Put LLM-generated synthetic queries only (do not use real user queries).

# Prepare data

## For generate preprocess data
The preprocessing pipeline is located under the `Train_model` directory and consists of the `preprocess_generate` and `preprocess_data` folders.

**1. Change to the Train_model/preprocess_generate directory.**
```
Train_model/
└── preprocess_data/
└── final_merge_data_250_label.xlsx
```
```
cd Train_model
cd preprocess_generate
```


**2. Place your dataset in the `preprocess_data` directory.**

Be sure that your file like this, the filename can be changed if necessary.

| 編號 | 擬真問句 | 標準問句 | 
|:------: |-----------|-------------|
| 1 | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 我想要了解如何申訴 | 
| 1 | 申訴流程該怎麼進行？ | 我想要了解如何申訴 | 
| 1 | 申訴的流程是什麼？ | 我想要了解如何申訴 | 



If your dataset is distributed across multiple `.xlsx` files, run `merged_original_data.py` to merge them into a single file.
The output file will be:

* **Samples for merged_original_data.xlsx**


| 編號 | 擬真問句 | 標準問句 | 
| :---: |-------------|-----|
| 編號1 | 擬真問句 | 標準問句 | 
| 1 | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 我想要了解如何申訴？ | 
| 2 | 我要申訴那個罰單，要填什麼資料？去哪裡填？ | 我想要了解如何申訴？ | 


If using `merged_original_data.xlsx` as the input file, run `merge_data.py` to generate `final_merge_data_250_label.xlsx`, where **250** indicates the number of intent categories. You may rename the output file according to the number of intent categories in your dataset.

* **Samples for final_merge_data_250_label.xlsx**


 | 編號 | 擬真問句 | 標準問句 | 
 | :---: |-------------|-----|
 | 1 | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 我想要了解如何申訴？ | 
 | 1 | 申訴流程該怎麼進行？ | 我想要了解如何申訴？ | 
 | 1 | 申訴的流程是什麼？ | 我想要了解如何申訴？ | 



**3. Run `cutallcolumns.py` to extract the standard questions, user queries, and corresponding labels, generating `after_center_usual_question.xlsx`.**

* **Samples for after_center_usual_question.xlsx**

| QUESTION | SAMPLE | LABEL | 
|-------------|-----| :---: |
| 我想要了解如何申訴？ | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 1 | 
| 我想要了解如何申訴？ | 申訴流程該怎麼進行？ | 1 | 
| 我想要了解如何申訴？ | 申訴的流程是什麼？ | 1 | 


**4. Run `preprocess.py`. A new directory named `data` will be created under `Train_model`, containing the following files:**

* **Samples for train.tsv**

 | QUESTION | SAMPLE | LABEL | 
 |-------------|-----| :---: |
 | 我收到逕裁吊扣牌照的裁決書，還需要到裁決中心辦理嗎？ | 我收到逕裁吊扣的裁決書，要去裁決中心？ | 58 | 
 | 黃線臨停會被拖吊嗎？ | 如果我停在黃線上，會不會被拖吊？ | 202 | 
 | 騎樓或人行道什麼情況可以停車？車停自家騎樓會被罰嗎？ | 那個…我車放在自家騎樓會不會罰款啊？ | 208 | 


**5. Run `cut_for_all_data_label.py` to produce the file all_data_label.tsv, which will also be saved in the `data` folder.**

* **Samples for all_data_label.tsv**

| SAMPLE | LABEL| 
|-----| :---: |
| 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 1 | 
| 申訴的流程是什麼 | 1 | 

> **Column Description**
> - **SAMPLE**: User query
> - **LABEL**: Corresponding label ID

**6. Run `cut_for_question_to_label.py` to generate `question_to_label.tsv`, which maps each standard question to its label. The output file will be saved in the `data` directory.**

* **Samples for question_to_label.tsv**

 | QUESTION | LABEL | 
 |-----| :---: |
 | 我想要了解如何申訴？ | 1 | 
 | 收到罰單後，我認為有問題，我可以怎麼辦？ | 2 | 
 | 罰單不服可以上網申訴嗎？ | 3 | 


**7. Run `split_class.py` to generate `new_all_data_label.tsv`. The output file will be saved in the `data` directory.**

* **Samples for new_all_data_label.tsv**

| SAMPLE | LABEL | CLASS | ORI_LABEL | ROUTE | 
|-----| :---: |-----| :---: | :---: |
| 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 29 | 我想要了解如何申訴？ | 1 | 1 | 
| 申訴流程該怎麼進行？ | 29 | 我想要了解如何申訴？ | 1 | 1 | 
| 申訴的流程是什麼？ | 29 | 我想要了解如何申訴？ | 1 | 1 | 


**After completing the preprocessing pipeline, the `data` directory will contain:**
>* all_data_label.tsv : all data
>* question_to_label.tsv : all questions mapping to labels
>* train.tsv
>* val.tsv
>* test.tsv


# Training (For CL)

1. If it is the first time you execute, please build the **"model"** and **"result"** directories first.

```
mkdir model
mkdir result
```

2. Change to the Train_model directory.
```
cd Train_model
```

3. The file **"hyperparameter.py"** records the parameters we use for training. All parameter descriptions are annotated after the parameter.

* **Task Setting**
```
'data_dir': '../data/', # data directory (load path)
'model_dir': '../model/', # save model directory
'model_name': 'CL',
'model_type': 'CLBERT',
```

* **General Setting**
```
'has_val': True, # if there are validation datasets (True means including train/val/test)
'time': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), # annotate start training time
'seed': 1111, # random seed and model shuffle
'save_best': True, # if True, model will be saved according to acc(CE) / loss (CL); else model will be saved until last epoch of training
```

* **BERT Setting**
```
'config': 'hfl/chinese-roberta-wwm-ext', # BERT pre-trained config.
'max_len': 30, # the max length of input tokens
```

* **Specified Model Setting**
```
'train_objective': 'CL', 
'label_col': 'LABEL',
```

* **Hyperparameters**
```
'optimizer': 'Adam', # 'Adam', 
'lr': 1e-5, # the learning rate 5.5E-5
'epochs': 400, # training epochs
'batch_size': 256, # batch size, depend on your GPU
'dropout': 0.3, # how random amount will be give up
'num_class':104849, #Number of training samples 
'temperature': 0.1, # contrastive loss scale controll
```

4. Execute the **"run_train.py"** script to train the model. Start by running the CL component. If you only intend to use the CL model, the training process is complete at this stage. To train the CLCE model, continue by executing the MORE (CLCE) component after the CL training has finished.

 Upon successful completion of the training, a new directory named with the format `CL_YYYY-MM-DD` (e.g., `CL_2026-07-06`) will be automatically created inside the `model` folder. This directory will contain the following Hugging Face compatible model and tokenizer files:

```text
model/
└── CL_YYYY-MM-DD/
       ├── config.json
       ├── model.safetensors
       ├── special_tokens_map.json
       ├── tokenizer_config.json
       ├── tokenizer.json
       └── vocab.txt
```


# Student model
Switch to the `student` directory.
Put only real user queries (do not use LLM-generated synthetic queries).
**Note:** The following **data preprocessing** and **model training** procedures are identical to those used for the **Teacher model**.

# Prepare data

## For generate preprocess data
The preprocessing pipeline is located under the `Train_model` directory and consists of the `preprocess_generate` and `preprocess_data` folders.

**1. Change to the Train_model/preprocess_generate directory.**
```
Train_model/
└── preprocess_data/
└── final_merge_data_250_label.xlsx
```
```
cd Train_model
cd preprocess_generate
```


**2. Place your dataset in the `preprocess_data` directory.**

Be sure that your file like this, the filename can be changed if necessary.

| 編號 | 擬真問句 | 標準問句 | 
|:------: |-----------|-------------|
| 1 | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 我想要了解如何申訴 | 
| 1 | 申訴流程該怎麼進行？ | 我想要了解如何申訴 | 
| 1 | 申訴的流程是什麼？ | 我想要了解如何申訴 | 



If your dataset is distributed across multiple `.xlsx` files, run `merged_original_data.py` to merge them into a single file.
The output file will be:

* **Samples for merged_original_data.xlsx**


| 編號 | 擬真問句 | 標準問句 | 
| :---: |-------------|-----|
| 編號1 | 擬真問句 | 標準問句 | 
| 1 | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 我想要了解如何申訴？ | 
| 2 | 我要申訴那個罰單，要填什麼資料？去哪裡填？ | 我想要了解如何申訴？ | 


If using `merged_original_data.xlsx` as the input file, run `merge_data.py` to generate `final_merge_data_250_label.xlsx`, where **250** indicates the number of intent categories. You may rename the output file according to the number of intent categories in your dataset.

* **Samples for final_merge_data_250_label.xlsx**

 | 編號 | 擬真問句 | 標準問句 | 
 | :---: |-------------|-----|
 | 1 | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 我想要了解如何申訴？ | 
 | 1 | 申訴流程該怎麼進行？ | 我想要了解如何申訴？ | 
 | 1 | 申訴的流程是什麼？ | 我想要了解如何申訴？ | 



**3. Run `cutallcolumns.py` to extract the standard questions, user queries, and corresponding labels, generating `after_center_usual_question.xlsx`.**

* **Samples for after_center_usual_question.xlsx**

| QUESTION | SAMPLE | LABEL | 
|-------------|-----| :---: |
| 我想要了解如何申訴？ | 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 1 | 
| 我想要了解如何申訴？ | 申訴流程該怎麼進行？ | 1 | 
| 我想要了解如何申訴？ | 申訴的流程是什麼？ | 1 | 


**4. Run `preprocess.py`. A new directory named `data` will be created under `Train_model`, containing the following files:**

* **Samples for train.tsv**

 | QUESTION | SAMPLE | LABEL | 
 |-------------|-----| :---: |
 | 我收到逕裁吊扣牌照的裁決書，還需要到裁決中心辦理嗎？ | 我收到逕裁吊扣的裁決書，要去裁決中心？ | 58 | 
 | 黃線臨停會被拖吊嗎？ | 如果我停在黃線上，會不會被拖吊？ | 202 | 
 | 騎樓或人行道什麼情況可以停車？車停自家騎樓會被罰嗎？ | 那個…我車放在自家騎樓會不會罰款啊？ | 208 | 


**5. Run `cut_for_all_data_label.py` to produce the file all_data_label.tsv, which will also be saved in the `data` folder.**

* **Samples for all_data_label.tsv**

| SAMPLE | LABEL| 
|-----| :---: |
| 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 1 | 
| 申訴的流程是什麼 | 1 | 

> **Column Description**
> - **SAMPLE**: User query
> - **LABEL**: Corresponding label ID

**6. Run `cut_for_question_to_label.py` to generate `question_to_label.tsv`, which maps each standard question to its label. The output file will be saved in the `data` directory.**

* **Samples for question_to_label.tsv**

 | QUESTION | LABEL | 
 |-----| :---: |
 | 我想要了解如何申訴？ | 1 | 
 | 收到罰單後，我認為有問題，我可以怎麼辦？ | 2 | 
 | 罰單不服可以上網申訴嗎？ | 3 | 


**7. Run `split_class.py` to generate `new_all_data_label.tsv`. The output file will be saved in the `data` directory.**

* **Samples for new_all_data_label.tsv**

| SAMPLE | LABEL | CLASS | ORI_LABEL | ROUTE | 
|-----| :---: |-----| :---: | :---: |
| 欸我剛收到一張罰單，我想問一下這個我可以怎麼申訴？ | 29 | 我想要了解如何申訴？ | 1 | 1 | 
| 申訴流程該怎麼進行？ | 29 | 我想要了解如何申訴？ | 1 | 1 | 
| 申訴的流程是什麼？ | 29 | 我想要了解如何申訴？ | 1 | 1 | 


**After completing the preprocessing pipeline, the `data` directory will contain:**
>* all_data_label.tsv : all data
>* question_to_label.tsv : all questions mapping to labels
>* train.tsv
>* val.tsv
>* test.tsv


# Training (For CL)

1. Copy the trained CL model generated from the **Teacher model** into the `model` directory.

The directory structure should look like:

```text
model/
└── CL_YYYY-MM-DD/
       ├── config.json
       ├── model.safetensors
       ├── special_tokens_map.json
       ├── tokenizer_config.json
       ├── tokenizer.json
       └── vocab.txt
```

This model will be loaded as the initialization checkpoint for training the Student model.

2. Change to the Train_model directory.
```
cd Train_model
```

3. The file **"hyperparameter.py"** records the parameters we use for training. All parameter descriptions are annotated after the parameter.

* **Task Setting**
```
'data_dir': '../data/', # data directory (load path)
'model_dir': '../model/', # save model directory
'model_name': 'CL',
'model_type': 'CLBERT',
```

* **General Setting**
```
'has_val': True, # if there are validation datasets (True means including train/val/test)
'time': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), # annotate start training time
'seed': 1111, # random seed and model shuffle
'save_best': True, # if True, model will be saved according to acc(CE) / loss (CL); else model will be saved until last epoch of training
```

* **BERT Setting**
```
'config': 'hfl/chinese-roberta-wwm-ext', # BERT pre-trained config.
'max_len': 30, # the max length of input tokens
```

* **Specified Model Setting**
```
'train_objective': 'CL', 
'label_col': 'LABEL',
```

* **Hyperparameters**
```
'optimizer': 'Adam', # 'Adam', 
'lr': 1e-5, # the learning rate 5.5E-5
'epochs': 50, # training epochs
'batch_size': 256, # batch size, depend on your GPU
'dropout': 0.3, # how random amount will be give up
'num_class':1447, #If 'model_type'='CLBERT', 'num_class'=Number of training samples 
'temperature': 0.1, # contrastive loss scale controll
'ckpt': 'CL_YYYY-MM-DD', # Load Teacher CL model
```

4. Execute the **"run_train.py"** script to train the model. Start by running the CL component. If you only intend to use the CL model, the training process is complete at this stage. To train the CLCE model, continue by executing the MORE (CLCE) component after the CL training has finished.

 Upon successful completion of the training, a new directory named with the format `CL_YYYY-MM-DD` (e.g., `CL_2026-07-06`) will be automatically created inside the `model` folder. This directory will contain the following Hugging Face compatible model and tokenizer files:

```text
model/
└── CL_YYYY-MM-DD/
       ├── config.json
       ├── model.safetensors
       ├── special_tokens_map.json
       ├── tokenizer_config.json
       ├── tokenizer.json
       └── vocab.txt
```



## Inference 

Please first confirm that the model is located in the `model` folder. 
Then switch the environment to the `AC_KG_CL_Ranker` folder.

```
cd AC_KG_CL_Ranker
```

Before starting, prepare the following three materials.
1. Verification Question `驗證題.xlsx`

| text | true_label | true_class |
|-------------| :---: |-----|
| 先生，你好，我有收到1張民眾檢舉的那個名單，那我如果要去申訴的話，是到你們現場嗎？ | 1 | 我想要了解如何申訴？| 


2. Keywords `關鍵字.xlsx`

| 知識編號 | 標準問句 | 關鍵字 |
| :---: |-------------|-----|
| 1 | 我想要了解如何申訴？ | `['處理方式', '投訴', '有問題', '不服', '紅單', '方式', '罰單', '管道', '陳訴', '方法', '不合理', '申訴', '如何']` |

3. The number corresponds to the standard question `編號對應標準問句.xlsx`


| 知識編號 | 標準問句 | 
| :---: |-------------|
| 1 | 我想要了解如何申訴？ |


Before running inference, first determine which retrieval strategy and enhancement method you would like to use.

## Available Scripts

The following scripts are used during inference:

| Script | Description |
|---------|-------------|
| `many_query_run_excel.py` | Knowledge Graph → Contrastive Learning retrieval |
| `many_query_run_excel_twotype_hybrid_top3.py` | Bidirectional Hybrid retrieval |
| `build_graph_pkl.py` | Build the Knowledge Graph `.pkl` file (required when deploying the API) |

## Retrieval Strategies

Two retrieval strategies are provided.

### Strategy 1. Knowledge Graph → Contrastive Learning `many_query_run_excel.py`

1. All standard questions are indexed into the Knowledge Graph.
2. The Knowledge Graph retrieves the top-*k* candidate standard questions.
3. The top-*k* candidates are passed to the Contrastive Learning (CL) model.
4. If the highest CL confidence exceeds `RERANK_THRESHOLD`, the top-1 prediction is directly returned.
5. Otherwise, the final prediction is reranked using the Hybrid Score.

$$Score_{final}(q_i) = \alpha \times Score_{CL}(q_i) + (1-\alpha) \times Score_{AC}(q_i)$$


### Strategy 2. Bidirectional Hybrid Retrieval `many_query_run_excel_twotype_hybrid_top3.py`

1. All standard questions are indexed into the Knowledge Graph.
2. The Knowledge Graph retrieves the top-*m* candidates.
3. The CL model retrieves the top-*n* candidates from all standard questions.
4. The union of the two candidate sets is used as the final top-*k* candidates.
5. Hybrid Score is applied to rerank the candidates and generate the final top-1 prediction.

$$Score_{final}(q_i) = \alpha \times Score_{CL}(q_i) + (1-\alpha) \times Score_{AC}(q_i)$$

## Enhancement Methods

Two optional enhancement methods are provided.

### 1. Keyword Weighting

Specific keywords (e.g., **"申訴"**, **"不合理"**, **"不服"**, **"有問題"**) receive additional weights during Knowledge Graph retrieval.

### 2. Hard-wired Standard Question Weighting

Certain standard questions can receive additional scores when predefined keywords appear.

For example,

- Keywords:
  - 申訴
  - 不合理

- Standard Question IDs:
  - 1
  - 2

If one of these keywords appears,

- If the corresponding standard question is **not** in the candidate list, it will be inserted into Top-*k* with an additional score.
- If it is already in the candidate list, its score will be further increased.


# Configuration

## Input Files

Before running inference, modify the following file paths if necessary.

```python
label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句檔案")
query_df = pd.read_excel(CURRENT_DIR / "1030_驗證題.xlsx")
builder.load_data_from_excel(CURRENT_DIR / "關鍵字檔案")
MODEL_PATH = ROOT_DIR / "model" / "模型"
```

## Hyperparameters

| Parameter | Description |
|-----------|-------------|
| `AC_TOPN_LIST` | Number of candidate standard questions retained by the Knowledge Graph retrieval. |
| `CL_TOPN_LIST` | Number of candidate standard questions retained by the Contrastive Learning model. |
| `NEIGHBOR_WEIGHT_LIST` | Weight assigned to neighboring keywords in the Knowledge Graph. Direct keyword matches always receive a score of 1. |
| `RERANK_THRESHOLD_LIST` | Confidence threshold for enabling Hybrid Score reranking. |
| `ALPHA_LIST` | Weight α for balancing CL score and Knowledge Graph score. |
| `MU_LIST` | Similarity threshold used when constructing the Knowledge Graph. |
| `KEYWORD_THRESHOLD_LIST` | Minimum number of matched keywords required to enable keyword position weighting. |
| `WEIGHT_STRATEGY_LIST` | Strategy for increasing weights when keywords appear near the end of the sentence. |
| `GAMMA_LIST` | Penalty exponent for long standard questions. |
| `SELECTED_KEYWORDS` | List of predefined keywords. |
| `SPECIAL_MAIN_WEIGHT_LIST` | Extra weight assigned to direct matches of predefined keywords. |
| `SPECIAL_NEIGHBOR_WEIGHT_LIST` | Extra weight assigned to neighboring predefined keywords. |
| `SPECIAL_STANDQ_ID_LIST` | IDs of hard-wired standard questions. |
| `SPECIAL_STANDQ_WEIGHT_LIST` | Additional score assigned to hard-wired standard questions. |
