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

def process_ex_score(bed_root, extract_num_list, fold_num, in_fold_num):
    # yi 마다의 linear 값 계산
    for y_value in ['y', 'y1', 'y2', 'y3']:
        df_linear = pd.read_csv(get_bed_path('linear_yi_assoc', 'train', fold_num, in_fold_num=in_fold_num,  y=y_value) , delim_whitespace=True, error_bad_lines=False)
        df_linear = df_linear[df_linear['TEST']=='ADD']
        df_linear = df_linear[df_linear['P'] != 'NA']  # P가 계산되지 않은 경우 제외
        df_linear = df_linear[df_linear['OR'] != 'NA']  # BETA가 계산되지 않은 경우 제외
        df_linear['LOR'] = df_linear['OR'].apply(np.log)  # Log OR로 변경    

        for ex_num in extract_num_list:    
            df_linear_ex = df_linear.nsmallest(ex_num, 'P')[['SNP', 'A1', 'LOR']]  # P-value 기준으로 추출
            df_linear_ex.to_csv(get_bed_path('linear_yi_extract', 'train', fold_num, in_fold_num=in_fold_num, y=y_value, ex=ex_num), index=False, header=False, sep='\t')

            # score 계산
            for score_train_type in ['train', 'test']:
                execute_cmd(f"plink --bfile {get_bed_path('keep_bed', score_train_type, fold_num, in_fold_num=in_fold_num)}"\
                    + f" --score {get_bed_path('linear_yi_extract', 'train', fold_num, in_fold_num=in_fold_num, y=y_value, ex=ex_num)} sum"\
                    + f" --out {get_bed_path('linear_yi_score', score_train_type, fold_num, in_fold_num=in_fold_num, y=y_value, ex=ex_num)}")
        

if __name__ == "__main__":
    
# 분석 변수 설정
    bed_root = '/home2/jpark/Projects/prs/data/bed'
    extract_num_list = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 5000, 7500,
                        10000, 20000, 30000, 50000] 
    random_seed = 42
    n_jobs = 13

# score 계산 수행
    k_fold_product = product(range(1, 5 + 1), [None] + list(range(1, 5 + 1)))
    with parallel_backend('loky', n_jobs=n_jobs):
        Parallel()(delayed(process_ex_score)(bed_root, extract_num_list, k_fold, in_k_fold) for k_fold, in_k_fold in k_fold_product)
