from sklearn.preprocessing import StandardScaler

model_root = '/home2/jpark/Projects/prs/model'
bed_root = '/home2/jpark/Projects/prs/data/bed'

def normalize(data, col, fit_scaler=None):    
    if fit_scaler is None:
        scaler = StandardScaler()
        fit_scaler = scaler.fit(data[[col]])
    data[[col]] = fit_scaler.transform(data[[col]])
    return fit_scaler

def get_bed_path(data_type, train_type, fold_num, in_fold_num=None, y=None, ex=None):
    cv_path = f'{bed_root}/bed_cv2/cv_{fold_num}'
    if in_fold_num is None:
        cv_num = f'{fold_num}'
    else:
        cv_num = f'{fold_num}_{in_fold_num}'

    if train_type not in ('train', 'test'):
        raise Exception(f'train_type: {train_type}')
    
    if data_type == 'score':
        return_path = f"{get_bed_path('linear_yi', train_type, fold_num, in_fold_num, y, ex)}_{train_type}_score.profile"
    elif data_type == 'label':
        return_path = f'{cv_path}/cv_{cv_num}_{train_type}_norm.tsv'
    elif data_type == 'linear_yi':
        if ex == None:
            return_path = f'{cv_path}/cv_{cv_num}_train_linear_{y}'
        else:
            return_path = f'{cv_path}/cv_{cv_num}_train_linear_{y}_{ex}'
    elif data_type == 'linear_yi_assoc':
        return_path = f"{get_bed_path('linear_yi', train_type, fold_num, in_fold_num, y)}.assoc.logistic"
    elif data_type == 'linear_yi_extract':
        return_path = f"{get_bed_path('linear_yi', train_type, fold_num, in_fold_num, y, ex)}_extract.tsv"
    elif data_type == 'linear_yi_extract_snp':
        return_path = f"{get_bed_path('linear_yi', train_type, fold_num, in_fold_num, y, ex)}_extract_snp.tsv"
    elif data_type == 'linear_yi_score':
        return_path = f"{get_bed_path('linear_yi', train_type, fold_num, in_fold_num, y, ex)}_{train_type}_score"
    elif data_type == 'keep_bed':
        return_path = f'{cv_path}/temp/cv_{cv_num}_{train_type}_keep'
    elif data_type == 'keep_bed_yi_ex':
        return_path = f'{cv_path}/temp/cv_{cv_num}_{train_type}_keep_{y}_{ex}'
    elif data_type == 'model_yi_ex':
        return_path = f'{model_root}/cv_{cv_num}_{train_type}_{y}_{ex}'
    else:
        raise Exception(f'data_type: {data_type}')
    
    return return_path
