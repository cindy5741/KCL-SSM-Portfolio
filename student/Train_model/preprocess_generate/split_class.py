import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# get question <-> label mapping
def map_question_label(fname=None, df=None):
    q2l, l2q = {}, {}
    if fname and df is None:
        all_data = pd.read_csv(fname, sep='\t')
    elif fname is None and df is not None:
        all_data = df
    for i in range(len(all_data)):
        q = all_data['QUESTION'][i]
        l = all_data['LABEL'][i]
        if q not in q2l:
            q2l[q] = l
            l2q[l] = q
    return q2l, l2q

# record the acuumulation of each label
def statistics(fname):
    all_data = pd.read_csv(fname, sep='\t')
    q_count = {}
    for i in range(len(all_data)):
        if all_data['LABEL'][i] not in q_count:
            q_count[all_data['LABEL'][i]] = 0
        q_count[all_data['LABEL'][i]] += 1
    return all_data, q_count

# get the more-class & less-class respectivaly
def split_class_by_num(data_fname, all_q_fname=None, num=10):

    all_df, q_count = statistics(data_fname)
    if all_q_fname is not None:
        _, all_l2q = map_question_label(fname=all_q_fname)
    else:
        _, all_l2q = map_question_label(df=all_df)
    new_data = []
    for i in range(len(all_df)):
        l = all_df['LABEL'][i]
        c = all_l2q[l]
        s = all_df['SAMPLE'][i]
        new_data.append({'SAMPLE': s, 'LABEL':l, 'CLASS': c, 'is_lower': q_count[l]<num})
    new_df = pd.DataFrame(new_data, columns=['SAMPLE', 'LABEL', 'CLASS', 'is_lower'])

    less = new_df[new_df['is_lower']==True].CLASS.unique()
    more = new_df[new_df['is_lower']==False].CLASS.unique()

    return all_df, less, more

# use input data and re-build label index
def build_new_question_to_label(data, data_dir):
    q2l, l2q = {}, {} # question to label; label to question
    random.seed(1111)
    random.shuffle(data) # shffule question and define label
    for idx, q in enumerate(data):
        q2l[q] = idx
        l2q[idx] = q

    # save the new quesiotn-label file
    save_dict = []
    for q, l in q2l.items():
        save_dict.append({'QUESTION': q, 'LABEL': l})
    ql_df = pd.DataFrame(save_dict, columns=['QUESTION', 'LABEL'])
    # ql_df.to_csv(data_dir+'question_to_label.tsv', sep='\t', index=False) # Using Tab as delimiter

    return ql_df

# build new data
def build_data(all_df, all_q_fname, more_q, less_q):
    '''
    all_df: pd.DataFrame, (Format: 'SAMPLE', 'LABEL')
    all_q_fname: str, (Format: 'QUESTION', 'LABEL')
    more_q: pd.DataFrame, (Format: 'QUESTION', 'LABEL')
    less_q: pd.DataFrame, (Format: 'QUESTION', 'LABEL')
    '''
    more_q2l, more_l2q = map_question_label(df=more_q) # get quesiotn-label dictionary of "more class" data
    less_q2l, less_l2q = map_question_label(df=less_q) # get quesiotn-label dictionary of "less class" data
    all_q2l, all_l2q = map_question_label(fname=all_q_fname)

    new_data = []
    for i in range(len(all_df)):
        s = all_df['SAMPLE'][i] # input query
        l = all_df['LABEL'][i] # origin label
        c = all_l2q[l] # label name
        if c in less_q2l.keys():
            new_l = less_q2l[c]
            new_data.append({'SAMPLE':s, 'LABEL':new_l, 'CLASS': c, 'ORI_LABEL':l, 'ROUTE':0})
        else:
            new_l = more_q2l[c]
            new_data.append({'SAMPLE':s, 'LABEL':new_l, 'CLASS': c, 'ORI_LABEL':l, 'ROUTE':1})
    new_df = pd.DataFrame(new_data, columns=['SAMPLE', 'LABEL', 'CLASS', 'ORI_LABEL', 'ROUTE'])
    '''
    # column descriptions #
    "SAMPLE" -> user query, 
    "LABEL" -> classification label, 
    "CLASS" -> corresponding label class name,
    "ORI_LABEL" -> the original label without spliting, 
    "ROUTE": 'less' belong to route '0'; 'more' means route '1',
    '''
    return new_df

# split data 
def split_data(df, save_dir, has_val=True):
    '''
    df: pd.DataFrame, the data after preprocessing
    save_dir: str, the data directory path
    has_val: boolean, if True, split data into train/val/test (with 80%/10%/10%); else train/test (with 90%/10%)
    '''

    if save_dir[-1] != '/':
        save_dir = save_dir +'/'
        
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
   

    if not has_val:
        d_train, d_test = train_test_split(df, random_state=1111, train_size=0.9, test_size=0.1) # 90% for training, 10% for testing
        print('# of train data:', len(d_train))
        print('# of test data:', len(d_test))
    else:
        d_train, d_temp = train_test_split(df, random_state=1111, train_size=0.8, test_size=0.2) # 80% for training
        d_val, d_test = train_test_split(d_temp, random_state=1111, train_size=0.5, test_size=0.5) # 10% for validation, 10% for testing
        print('# of train data:', len(d_train))
        print('# of val data:', len(d_val))
        print('# of test data:', len(d_test))

    d_train.to_csv(save_dir + 'train.tsv', sep='\t', index=False)
    d_test.to_csv(save_dir + 'test.tsv', sep='\t', index=False)
    if has_val:
        d_val.to_csv(save_dir + 'val.tsv', sep='\t', index=False)

if __name__ == '__main__':

    INPUT_DATA_DIR = '../../data/' # the directory of original input file
    # the following two files are paired
    ALL_ORIGIN_DATA = 'all_data_label.tsv' # the format is 'SAMPLE' and 'LABEL'
    ALL_ORIGIN_Q2L = 'question_to_label.tsv' # the format is 'QUESTION' and 'LABEL'
    
    LIMIT_NUM = 2 # the number for dividing class
    OUTPUT_LESS_DIR = '../../data/less/' # the class which less than num [LABEL]
    OUTPUT_MORE_DIR = '../../data/more/' # the class which greater (included) than num [LABEL]
    OUTPUT_ROUTE_DIR = '../../data/route/' # the samples which should generate to stage I or II [RELATION]
    
    # check directory exists
    # if not os.path.exists(OUTPUT_LESS_DIR):
    #     os.makedirs(OUTPUT_LESS_DIR)
    # if not os.path.exists(OUTPUT_MORE_DIR):
    #     os.makedirs(OUTPUT_MORE_DIR)
    # if not os.path.exists(OUTPUT_ROUTE_DIR):
    #     os.makedirs(OUTPUT_ROUTE_DIR)

    # split original data into two pieces
    all_df, less, more = split_class_by_num(INPUT_DATA_DIR + ALL_ORIGIN_DATA, 
                                         all_q_fname = INPUT_DATA_DIR + ALL_ORIGIN_Q2L, 
                                         num=LIMIT_NUM) 

    # build the corresponding corpus of two phases
    less_q_df = build_new_question_to_label(less, OUTPUT_LESS_DIR)
    more_q_df = build_new_question_to_label(more, OUTPUT_MORE_DIR)

   
    # Merge all labels into a single class called 'more'
    more_q_df = build_new_question_to_label(all_df['LABEL'].map(
        lambda l: map_question_label(fname=INPUT_DATA_DIR + ALL_ORIGIN_Q2L)[1][l]
    ).unique(), OUTPUT_MORE_DIR)

    
    less_q_df = pd.DataFrame(columns=['QUESTION', 'LABEL'])

    
    # build new data (Format: 'SAMPLE', 'LABEL', 'CLASS', 'ORI_LABEL', 'RELATION')
    new_df = build_data(all_df, INPUT_DATA_DIR + ALL_ORIGIN_Q2L, more_q_df, less_q_df)
    new_df.to_csv(INPUT_DATA_DIR + 'new_all_data_label.tsv', sep='\t', index=False) # Using Tab as delimiter
    
    # filter the more/less new data
    less_df = new_df[new_df['ROUTE']==0]
    more_df = new_df[new_df['ROUTE']==1]
    print('# of data that class more :', len(more_df))
    
   

