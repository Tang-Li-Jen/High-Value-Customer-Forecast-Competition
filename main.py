#!/usr/bin/env python
# coding: utf-8

# In[314]:


from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import os


# In[315]:


folder_path = 'data/'


# In[316]:


os.listdir(folder_path)


# In[345]:


df_month = pd.read_csv(os.path.join(folder_path, 'month_amount.csv'))
df_feature_v5 = pd.read_csv(os.path.join(folder_path, 'feature_v5.csv'))


# In[317]:


df_feature_v4 = pd.read_csv(os.path.join(folder_path, 'feature_v4.csv'))


# In[298]:


df_feature_v3 = pd.read_csv(os.path.join(folder_path, 'feature_v3.csv'))


# In[248]:


df_growth_features = pd.read_csv(os.path.join(folder_path, 'growth_features.csv'))


# In[216]:


df_rfm_features = pd.read_csv(os.path.join(folder_path, 'rfm_features.csv'))


# In[217]:


df_train = pd.read_csv(os.path.join(folder_path, 'user_label_train.csv'))
df_test = pd.read_csv(os.path.join(folder_path, 'submission.csv'))

df_user_info = pd.read_csv(os.path.join(folder_path, 'user_info.csv'))
df_login = pd.read_csv(os.path.join(folder_path, 'login.csv'))
df_purchase_detail = pd.read_csv(os.path.join(folder_path, 'purchase_detail.csv'))


# In[218]:


df_train['label'].value_counts(normalize=True)


# In[219]:


print('the number of train: {}'.format(len(df_train)))
print('the number of test: {}'.format(len(df_test)))


# ## Feature Engineering

# In[220]:


df_train = pd.merge(df_train, df_user_info, on='userid', how='inner')
df_test = pd.merge(df_test, df_user_info, on='userid', how='inner')

print('the number of train: {}'.format(len(df_train)))
print('the number of test: {}'.format(len(df_test)))


# In[221]:


df_train['age'] = 2020 - df_train.birth_year
df_test['age'] = 2020 - df_test.birth_year


# In[222]:


df_train['lifetime'] = pd.to_datetime('2020-07-31') - pd.to_datetime(df_train['enroll_time'])
df_train.lifetime = df_train.lifetime.astype('timedelta64[D]')

df_test['lifetime'] = pd.to_datetime('2020-07-31') - pd.to_datetime(df_test['enroll_time'])
df_test.lifetime = df_test.lifetime.astype('timedelta64[D]')


# In[223]:


df_login_feature = df_login.groupby('userid').agg({'login_times':['sum', 'min', 'max', 'std', 'mean']})
df_login_feature.columns = ["_".join(x) for x in df_login_feature.columns.ravel()]


# In[224]:


df_train = pd.merge(df_train, df_login_feature, on='userid', how='inner')
df_test = pd.merge(df_test, df_login_feature, on='userid', how='inner')


# In[342]:


df_dt = df_purchase_detail[['userid','date']].drop_duplicates()


# In[338]:


df_purchase_detail = df_purchase_detail.sort_values(['userid','date'], ascending=[1,1])


# In[330]:


df_purchase_detail['date'] = pd.to_datetime(df_purchase_detail.grass_date)


# In[343]:


df_dt['dt_diff'] = df_dt.groupby('userid')['date'].diff().astype('timedelta64[D]')


# In[360]:


df_purchase_dt_diff


# In[361]:


df_train = pd.merge(df_train, df_purchase_dt_diff, on='userid', how='inner')
df_test = pd.merge(df_test, df_purchase_dt_diff, on='userid', how='inner')


# In[359]:


df_purchase_dt_diff = df_dt.groupby('userid').agg({'dt_diff':['sum', 'min', 'max', 'std','mean']
                                                                      })
df_purchase_dt_diff.columns = ["_".join(x) for x in df_purchase_dt_diff.columns.ravel()]


# In[339]:


df_purchase_detail[df_purchase_detail.userid == 295790]


# In[ ]:





# In[300]:


df_train = pd.merge(df_train, df_feature_v3, on='userid', how='inner')
df_test = pd.merge(df_test, df_feature_v3, on='userid', how='inner')


# In[318]:


df_train = pd.merge(df_train, df_feature_v4, on='userid', how='inner')
df_test = pd.merge(df_test, df_feature_v4, on='userid', how='inner')


# In[346]:


df_train = pd.merge(df_train, df_feature_v5, on='userid', how='inner')
df_test = pd.merge(df_test, df_feature_v5, on='userid', how='inner')


# In[347]:


df_train = pd.merge(df_train, df_month, on='userid', how='inner')
df_test = pd.merge(df_test, df_month, on='userid', how='inner')


# In[263]:


df_purchase_ndays = df_purchase_detail.groupby('userid')['grass_date'].count()


# In[265]:


df_train = pd.merge(df_train, df_purchase_ndays, on='userid', how='inner')
df_test = pd.merge(df_test, df_purchase_ndays, on='userid', how='inner')


# In[274]:


df_purchase_detail['amount_per_order'] = df_purchase_detail.total_amount / df_purchase_detail.order_count


# In[226]:


df_purchase_detail_feature = df_purchase_detail.groupby('userid').agg({'order_count':['sum', 'min', 'max', 'std', 'mean']
                                                                      , 'total_amount':['sum', 'min', 'max', 'std','mean']})
df_purchase_detail_feature.columns = ["_".join(x) for x in df_purchase_detail_feature.columns.ravel()]


# In[227]:


df_train = pd.merge(df_train, df_purchase_detail_feature, on='userid', how='inner')
df_test = pd.merge(df_test, df_purchase_detail_feature, on='userid', how='inner')


# In[276]:


df_purchase_amount_per_order = df_purchase_detail.groupby('userid').agg({'amount_per_order':['sum', 'min', 'max', 'std','mean']
                                                                      })
df_purchase_amount_per_order.columns = ["_".join(x) for x in df_purchase_amount_per_order.columns.ravel()]


# In[277]:


df_train = pd.merge(df_train, df_purchase_amount_per_order, on='userid', how='inner')
df_test = pd.merge(df_test, df_purchase_amount_per_order, on='userid', how='inner')


# In[250]:


df_train = pd.merge(df_train, df_growth_features, on='userid', how='inner')
df_test = pd.merge(df_test, df_growth_features, on='userid', how='inner')


# In[228]:


df_rfm_features = df_rfm_features[['userid', 'DistinctDay', 'DistinctDayIn90days', 'DistinctDayIn60days',
       'DistinctDayIn30days', 'DistinctDayIn14days', 'DistinctDayIn7days',
       'DistinctDayIn3days', 'FreqIn90days', 'FreqIn60days',
       'FreqIn14days', 'FreqIn3days', 'FreqIn7days', 'rececny']]


# In[229]:


df_train = pd.merge(df_train, df_rfm_features, on='userid', how='inner')
df_test = pd.merge(df_test, df_rfm_features, on='userid', how='inner')


# In[230]:


df_category_count = df_purchase_detail.groupby(['userid','category_encoded'], as_index=False)['total_amount'].sum()

df_category_pivot = pd.pivot_table(df_category_count, index='userid', columns='category_encoded'
                                       , values='total_amount')

df_category_pivot = df_category_pivot.fillna(0)

df_category_pivot.columns = ['category_' + str(x) for x in df_category_pivot.columns]


# In[231]:


df_train = pd.merge(df_train, df_category_pivot, on='userid', how='inner')
df_test = pd.merge(df_test, df_category_pivot, on='userid', how='inner')


# ## Modeling

# In[362]:


df_train.columns


# In[363]:


df_tr, df_val = train_test_split(df_train, stratify = df_train['label'], test_size=0.2, random_state=42)


# In[364]:


features = ['gender', 'is_seller', 'birth_year', 'login_times_sum', 'login_times_min', 'login_times_max',
       'login_times_std', 'login_times_mean', 'age', 'lifetime', 'category_1', 'category_2',
       'category_3', 'category_4', 'category_5', 'category_6', 'category_7',
       'category_8', 'category_9', 'category_10', 'category_11', 'category_12',
       'category_13', 'category_14', 'category_15', 'category_16',
       'category_17', 'category_18', 'category_19', 'category_20',
       'category_21', 'category_22', 'category_23', 'DistinctDay', 'DistinctDayIn90days', 'DistinctDayIn60days',
       'DistinctDayIn30days', 'DistinctDayIn14days', 'DistinctDayIn7days',
       'DistinctDayIn3days', 'FreqIn90days', 'FreqIn60days',
       'FreqIn14days', 'FreqIn3days', 'FreqIn7days', 'rececny',
           'order_count_sum', 'order_count_min',
       'order_count_max', 'order_count_std','order_count_mean', 'total_amount_sum',
       'total_amount_min', 'total_amount_max', 'total_amount_std','total_amount_mean', 'avgFreqMoM','grass_date',
           'amount_per_order_sum',
       'amount_per_order_min', 'amount_per_order_max', 'amount_per_order_std',
       'amount_per_order_mean','AvgMoMOrderCnt', 'AvgMoMTotCnt',
       'OrderCntIn90days', 'TotCntIn90days', 'OrderCntIn60days',
       'TotCntIn60days', 'OrderCntIn30days', 'TotCntIn30days',
       'OrderCntIn14days', 'TotCntIn14days', 'OrderCntIn7days',
       'TotCntIn7days', 'OrderCntIn3days', 'TotCntIn3days', 'BuyRececny','DistinctCategory',
           'GoodBuyer', '2', '3', '4', '5', '6', '7', 'dt_diff_sum', 'dt_diff_min', 'dt_diff_max', 'dt_diff_std',
       'dt_diff_mean']

print(len(features))


# In[365]:


tr_X = df_tr[features].values
tr_y = df_tr['label']

val_X = df_val[features].values
val_y = df_val['label']

te_X = df_test[features].values


# In[366]:


lgtrain = lgb.Dataset(tr_X, tr_y)
lgvalid = lgb.Dataset(val_X, val_y)


# In[ ]:


# lgtrain = lgb.Dataset(tr_X, tr_y, categorical_feature=['sex','job_category','job_level','factory_id','mgt_level','current_pjt_role'
#                                                        ,'work_location','age_level','married','education_major','education_type','education'
#                                                       ,'department','commute_cost', 'new_comer','is_promotion','promotion_speed'
#                                                        ,'annual_perf_A', 'annual_perf_B', 'annual_perf_C','change_department'
#                                                        , 'new_comer','job_exp1', 'job_exp2', 'job_exp3','job_exp4', 'job_exp5'
#                                                       ])


# In[373]:


params = {
          "objective" : "binary",
          "num_leaves" : 30,
          "max_depth": -1,
          "bagging_fraction" : 0.8,  # subsample
          "feature_fraction" : 0.8,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "num_threads":4,
          'lambda_l1': 0.9, 
          'lambda_l2': 0.5, 
          'learning_rate': 0.01, 
          'metric': 'AUC',
          'is_unbalance': False,
          "verbosity" : -1 }


# In[374]:


evals_result = {}

clf = lgb.train(params
                , lgtrain
                , 1500
                , valid_sets=[lgvalid, lgtrain]
                , valid_names=['validation', 'train']
#                 , feval=lgb_fbeta_score
                , evals_result=evals_result
                ,early_stopping_rounds = 200,
                verbose_eval=50)


# ## Inference

# In[369]:


pred_test = clf.predict(te_X)


# In[370]:


sub = pd.read_csv(os.path.join(folder_path, 'submission.csv'))


# In[371]:


sub['label'] = pred_test


# In[372]:


sub.to_csv('submission_final.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




