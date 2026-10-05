from datetime import datetime

PARAM = {
    'data_dir': '../data/',
    'model_dir': '../model/',
    'model_name': 'CL_student',
    'model_type': 'CLBERT',

    'has_val': True,
    'time': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
    'seed': 1111,
    'save_best': True,

    'use_teacher_kd': True,
    'teacher_model_dir': '../model/',
    'teacher_ckpt': 'CL_teacher_2026-01-17',

    'alpha': 1,
    'beta': 1,
    'lambda': 0.9,

    'config': 'hfl/chinese-roberta-wwm-ext',

    # 避免 key 對不上
    'max_len': 30,
    'max_length': 30,

    'train_objective': 'CL',
    'label_col': 'LABEL',

    'optimizer': 'Adam',
    'lr': 1e-5,

    # 先保守
    'epochs': 50,
    'batch_size': 32,

    'dropout': 0.1,
    'temperature': 0.1,
    'num_class': 1447,

    'student_num_layers': 12,
    'init_student_from_teacher': True,
}
