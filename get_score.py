import numpy as np
import os
import pandas as pd
import time
import sys
from itertools import product
from joblib import parallel_backend, Parallel, delayed
from sklearn.model_selection import KFold

sys.path.append(os.path.abspath('/home2/jpark/Projects/prs/codes'))
from process import *


def execute_cmd(cmd):
    start_time = time.time()
    print(cmd)
    os.system(cmd)
    print(f'----- Execute cmd time(in minutes): {(time.time() - start_time)/60:.2f}m -----\n')

def process_score(bed_root, k_fold, in_k_fold):
    if in_k_fold is None:
        cv_train_name, cv_test_name = f'cv_{k_fold}_train', f'cv_{k_fold}_test'
    else:
        cv_train_name, cv_test_name = f'cv_{k_fold}_{in_k_fold}_train', f'cv_{k_fold}_{in_k_fold}_test'

    # cv path 설정 및 폴더 생성
    # bed 파일은 크기가 크므로, temp 폴더에 저장한 뒤 덮어씌운다
    cv_fold = f'{bed_root}/bed_cv2/cv_{k_fold}'
    cv_fold_temp = f'{cv_fold}/temp'
    execute_cmd(f'mkdir -p {cv_fold_temp}')  # bed_cv temp 폴더 생성

    # 파일 path 설정
    cv_train_tsv, cv_test_tsv = f'{cv_fold}/{cv_train_name}.tsv', f'{cv_fold}/{cv_test_name}.tsv'
    id_train_tsv, id_test_tsv = f'{cv_fold_temp}/{cv_train_name}_id.tsv', f'{cv_fold_temp}/{cv_test_name}_id.tsv'  #temp로만 활용
    cv_norm_train_tsv, cv_norm_test_tsv = f'{cv_fold}/{cv_train_name}_norm.tsv', f'{cv_fold}/{cv_test_name}_norm.tsv'
    chr_merged_bed = f'{bed_root}/bed/chr_merged'
    keep_train_bed, keep_test_bed = f'{cv_fold_temp}/{cv_train_name}_keep', f'{cv_fold_temp}/{cv_test_name}_keep'  #temp로만 활용
    linear_result = f'{cv_fold}/{cv_train_name}_linear'
    
    # cv 셋에 맞춰서 plink keep 수행
    k_train_df  = pd.read_csv(cv_train_tsv, sep='\t')
    k_test_df = pd.read_csv(cv_test_tsv, sep='\t')

    k_train_df.iloc[:, :2].to_csv(id_train_tsv, index=False, header=False, sep='\t')  # FID, IID 파일 생성
    k_test_df.iloc[:, :2].to_csv(id_test_tsv, index=False, header=False, sep='\t')  # FID, IID 파일 생성

    for col in ['age', 'bmi', 'sbp', 'dbp']:
        scaler = normalize(k_train_df, col)
        normalize(k_test_df, col, scaler)
    k_train_df.to_csv(cv_norm_train_tsv, index=False, sep='\t')
    k_test_df.to_csv(cv_norm_test_tsv, index=False, sep='\t')

    execute_cmd(f'plink -bfile {chr_merged_bed} --keep {id_train_tsv} --make-bed --out {keep_train_bed}')  # 해당 FID, IID로 구성된 bed 파일 생성
    execute_cmd(f'plink -bfile {chr_merged_bed} --keep {id_test_tsv} --make-bed --out {keep_test_bed}')  # 해당 FID, IID로 구성된 bed 파일 생성

    # yi 마다의 linear 값 계산
    for yi in ['y', 'y1', 'y2', 'y3']:
        linear_yi = f'{linear_result}_{yi}'
        linear_yi_assoc = f'{linear_yi}.assoc.logistic'
        linear_yi_extract_tsv = f'{linear_yi}_extract.tsv'
        linear_yi_train_score, linear_yi_test_score = f'{linear_yi}_train_score', f'{linear_yi}_test_score'
        execute_cmd(f'plink -bfile {keep_train_bed} --logistic --1 --pheno {cv_norm_train_tsv} --pheno-name {yi} --covar {cv_norm_train_tsv} --covar-name sex,age,bmi,sbp,dbp --allow-no-sex --out {linear_yi}')
        
        # p-value로 extract
#        df_linear = pd.read_csv(f'{linear_yi_assoc}', delim_whitespace=True, error_bad_lines=False)
#        df_linear = df_linear[df_linear['TEST']=='ADD']
#        df_linear = df_linear[df_linear['P'] != 'NA']  # P가 계산되지 않은 경우 제외
#        df_linear = df_linear[df_linear['OR'] != 'NA']  # BETA가 계산되지 않은 경우 제외
#        df_linear['LOR'] = df_linear['OR'].apply(np.log)  # Log OR로 변경
#        df_linear = df_linear[df_linear['P'] < extract_p_value][['SNP', 'A1', 'LOR']]  # P-value 기준으로 추출
#        df_linear.to_csv(f'{linear_yi_extract_tsv}', index=False, header=False, sep='\t')

        # score 계산
#        execute_cmd(f'plink --bfile {keep_train_bed} --score {linear_yi_extract_tsv} sum --out {linear_yi_train_score}')
#        execute_cmd(f'plink --bfile {keep_test_bed} --score {linear_yi_extract_tsv} sum --out {linear_yi_test_score}')

        # bed 파일은 용량이 크므로 바로 삭제
        #execute_cmd(f'rm {keep_train_bed}.bed')
        #execute_cmd(f'rm {keep_train_bed}.bim')
        #execute_cmd(f'rm {keep_train_bed}.fam')
        #execute_cmd(f'rm {keep_test_bed}.bed')
        #execute_cmd(f'rm {keep_test_bed}.bim')
        #execute_cmd(f'rm {keep_test_bed}.fam')

if __name__ == "__main__":
    
# 분석 변수 설정
    bed_root = '/home2/jpark/Projects/prs/data/bed'
    extract_p_value_list = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000]
    random_seed = 42
    n_jobs = 13

# CV 파일 생성
    df_result = pd.read_csv(f'{bed_root}/pheno_result.tsv', sep='\t')
    kf_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    k_fold = 1
    for train, test in kf_cv.split(df_result):
        df_train, df_test = df_result.iloc[train], df_result.iloc[test]
        execute_cmd(f'mkdir -p {bed_root}/bed_cv2/cv_{k_fold}')  # cv 폴더가 없으면 생성
        df_train.to_csv(f'{bed_root}/bed_cv2/cv_{k_fold}/cv_{k_fold}_train.tsv', index=False, sep='\t')
        df_test.to_csv(f'{bed_root}/bed_cv2/cv_{k_fold}/cv_{k_fold}_test.tsv', index=False, sep='\t')
        in_k_fold=1
        for in_train, in_test in kf_cv.split(df_train):  # train 에 해당하는 80%로 다시 5-fold CV 구성
            df_in_train, df_in_test = df_train.iloc[in_train], df_train.iloc[in_test]
            df_train.to_csv(f'{bed_root}/bed_cv2/cv_{k_fold}/cv_{k_fold}_{in_k_fold}_train.tsv', index=False, sep='\t')
            df_test.to_csv(f'{bed_root}/bed_cv2/cv_{k_fold}/cv_{k_fold}_{in_k_fold}_test.tsv', index=False, sep='\t')
            in_k_fold+=1
        k_fold+=1

# score 계산 수행
    k_fold_product = product(range(1, 5 + 1), [None] + list(range(1, 5 + 1)))
    with parallel_backend('loky', n_jobs=n_jobs):
        Parallel()(delayed(process_score)(bed_root, k_fold, in_k_fold) for k_fold, in_k_fold in k_fold_product)
