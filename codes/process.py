from sklearn.preprocessing import StandardScaler

model_root = '/home2/jpark/Projects/prs/model'
bed_root = '/home2/jpark/Projects/prs/data/bed'
pickle_root = '/home2/jpark/Projects/prs/pickle'

def normalize(data, col, fit_scaler=None):    
    if fit_scaler is None:
        scaler = StandardScaler()
        fit_scaler = scaler.fit(data[[col]])
    data[[col]] = fit_scaler.transform(data[[col]])
    return fit_scaler

def get_bed_path(data_type, train_type, fold_num, in_fold_num=None, y=None, ex=None, model_key=None, epoch=None):
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
        return_path = f'{model_root}/cv_{cv_num}_{train_type}_{y}_{ex}_{model_key}_{epoch}'
    elif data_type == 'hdict':
        return_path = f'{pickle_root}/hdict'
    else:
        raise Exception(f'data_type: {data_type}')
    
    return return_path

def get_epoch_min_loss(hdict, fold_num, in_fold_num): 
    min_loss = 1000
    min_epoch = 0
    for fold_i in hdict[0]:
        for epoch_test in [epoch_i[1] for epoch_i in fold_i]:  # test만 선택
            if epoch_test[1] == fold_num and epoch_test[2] == in_fold_num:
                if epoch_test[5] < min_loss:
                    min_loss = epoch_test[5]
                    min_epoch = epoch_test[0]
    return min_epoch