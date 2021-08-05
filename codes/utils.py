import pandas as pd
from IPython.core.display import HTML
from functools import reduce
from sklearn.preprocessing import StandardScaler


data_root = '/home2/jpark/Projects/prs/data'
bed_root = f'{data_root}/bed'

def execute_cmd(cmd):
    start_time = time.time()
    print(cmd)
    os.system(cmd)
    print(f'----- Execute cmd time(in minutes): {(time.time() - start_time)/60:.2f}m -----\n')

def check_kare_cdbk(df_cdbk, condition, as_period=range(1, 9),title=None, display_col=None, index=True, merge_period=False):
            
    if title is not None:
        display(HTML(f'<h3>{title}'))
    for period in as_period:
        if merge_period == False:
            df_as = df_cdbk[df_cdbk['테이블명_eng'].str.contains(f'AS{period}', case=False)]
        else:
            df_as = df_cdbk
        df_as = df_as.loc[condition]
        display(HTML(f'<h3>{period} 기'))
        if display_col is not None:
            df_as = df_as[display_col]
        if merge_period == True:
            df_as[display_col] = df_as[display_col].apply(lambda r: r[display_col].str.replace('유무', '여부'), axis=1)
            df_as = df_as.drop_duplicates()
        if index:
            display(df_as)
        else:
            display(HTML(df_as.to_html(index=False)))
            print('\n'.join([v[0] for v in df_as[display_col].values.tolist()]))
        if merge_period == True:
            break  # Merge 하는 경우 iter를 1회만 작동, display 케이스가 너무 다양하므로 향후 리팩토링 필요

            
def get_var_from_kare(df_cdbk, var, option='match', match_col='변수명', rename=None, display_cdbk=False, na_to_none=True):
    as_list = []
    # 다양한 조건에 맞춰 매칭
    if option == 'match':
        match_cond = df_cdbk[match_col].str.match(f'AS[0-9]_{var}')
    elif option == 'match_exact':
        match_cond = df_cdbk[match_col].str.match(var)
    elif option == 'end_exact':
        match_cond = df_cdbk[match_col].str.endswith(f'_{var}')
    elif option == 'contains':
        if type(var)==list:
            match_cond = reduce(lambda x, y: x & y, [df_cdbk[match_col].str.contains(v) for v in var])
        else:
            match_cond = df_cdbk[match_col].str.contains(var)
    else:
        raise NotImplementedError(f'option: {option}')
        
    matched_cdbk = df_cdbk[match_cond]
    if display_cdbk:
        display(df_cdbk[match_cond])
    for i, r in matched_cdbk.iterrows():
        as_num = r['변수명'].split('_')[0]
        table_df = pd.read_csv(f"{data_root}/{as_num}/{r['테이블명_eng']}.txt", encoding='EUCKR')[['RID', r['변수명']]]

        # NA 값을 처리
        if na_to_none:
            table_df[r['변수명']] = table_df\
                .apply(lambda row: None if int(row[r['변수명']]) in [66666, 77777, 99999] else row[r['변수명']],
                       axis=1)
        
        # AS기 마다 변수명이 다른 경우 Rename
        if rename is not None:
            table_df = table_df.rename(columns={r['변수명']: f'{as_num}_{rename}'})
        as_list.append(table_df)
    merged = reduce(lambda x, y: pd.merge(x, y, on='RID', how='outer'), as_list)
    return merged

def get_model_key(dict_input):
    return ''.join([f'{k}{v}' for k, v in dict_input.items()])