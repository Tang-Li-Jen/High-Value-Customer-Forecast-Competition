#!/usr/bin/env python
# coding: utf-8

## Import the packages
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from functools import reduce
import os

## Set the data folder path       
folder_path = '../data/'
#os.listdir(folder_path)
#-- current_path/
#   -- data/
#     -- login.csv
#     -- purchase_detail.csv
#     -- user_info.csv
#     -- user_label_train.csv

if __name__ == '__main__':
    ## Read the source data
    df_train = pd.read_csv(os.path.join(folder_path, 'user_label_train.csv'))
    df_test = pd.read_csv(os.path.join(folder_path, 'submission.csv'))
    df_user_info = pd.read_csv(os.path.join(folder_path, 'user_info.csv'))
    df_login = pd.read_csv(os.path.join(folder_path, 'login.csv'))
    df_purchase_detail = pd.read_csv(os.path.join(folder_path, 'purchase_detail.csv'))


    ## Read the result from features engineering
    #df_month = pd.read_csv(os.path.join(folder_path, 'month_amount.csv')) #Replaced by df_monthly_buy_sum_pivot
    df_feature_v3 = pd.read_csv(os.path.join(folder_path, 'feature_v3.csv'))
    df_feature_v4 = pd.read_csv(os.path.join(folder_path, 'feature_v4.csv'))
    df_feature_v5 = pd.read_csv(os.path.join(folder_path, 'feature_v5.csv'))
    df_growth_features = pd.read_csv(os.path.join(folder_path, 'growth_features.csv'))
    df_rfm_features = pd.read_csv(os.path.join(folder_path, 'rfm_features.csv'))


    # See the descriptive statistics of the data (could be used in ipynb)
    #df_train['label'].value_counts(normalize=True)
    #print('the number of train: {}'.format(len(df_train)))
    #print('the number of test: {}'.format(len(df_test)))


    ## Feature Engineering
    # Prepare the base dataset for training and testing
    df_train = pd.merge(df_train, df_user_info, on='userid', how='inner')
    df_test = pd.merge(df_test, df_user_info, on='userid', how='inner')

    #print('the number of train: {}'.format(len(df_train)))
    #print('the number of test: {}'.format(len(df_test)))

    # Process the age from the birth_year
    df_train['age'] = 2020 - df_train.birth_year
    df_test['age'] = 2020 - df_test.birth_year

    # Process the account lifetime to df_train and df_test respectively
    df_train['lifetime'] = pd.to_datetime('2020-07-31') - pd.to_datetime(df_train['enroll_time'])
    df_train.lifetime = df_train.lifetime.astype('timedelta64[D]')

    df_test['lifetime'] = pd.to_datetime('2020-07-31') - pd.to_datetime(df_test['enroll_time'])
    df_test.lifetime = df_test.lifetime.astype('timedelta64[D]')


    # Preliminarily Process the df_login (login.csv)
    # Process the login log to count total login times, min/max login times for a day, std and mean for overall login times
    df_login_feature = df_login.groupby('userid').agg({'login_times':['sum', 'min', 'max', 'std', 'mean']})
    df_login_feature.columns = ["_".join(x) for x in df_login_feature.columns.ravel()] #automatically add column names

    # Preliminarily Process the df_purchase_detail (purchase_detail.csv)
    # See the purchase detail of a certain user and know how data sorted
    #df_purchase_detail[df_purchase_detail.userid == 295790]

    # Process the field `grass_date` to be datatime type 
    df_purchase_detail['grass_date'] = pd.to_datetime(df_purchase_detail.grass_date)

    # Sort the purchase detail by userid and purchase date 
    df_purchase_detail = df_purchase_detail.sort_values(['userid', 'grass_date'], ascending=[1, 1])

    # Process the purchase time to see the freqency of a user's purchasement
    df_dt = df_purchase_detail[['userid','grass_date']].drop_duplicates()
    df_dt['dt_diff'] = df_dt.groupby('userid')['grass_date'].diff().astype('timedelta64[D]')
    df_purchase_dt_diff = df_dt.groupby('userid').agg({'dt_diff':['sum', 'min', 'max', 'std', 'mean']})
    df_purchase_dt_diff.columns = ["_".join(x) for x in df_purchase_dt_diff.columns.ravel()]
    #df_purchase_dt_diff.head()

    # Calculate the monthly purchase summary for each user (Replace original df_month)
    df_purchase_detail['month'] = df_purchase_detail.grass_date.dt.month

    df_monthly_buy = df_purchase_detail.groupby(['userid', 'month']).agg({'order_count':['sum', 'mean'], 'total_amount':['sum', 'mean']})
    df_monthly_buy.columns = ["_".join(x) for x in df_monthly_buy.columns.ravel()]

    # Transform the monthly summary into pivot table to easily merge back to df_train and df_train
    df_monthly_buy_sum_pivot = pd.pivot_table(df_monthly_buy, index='userid', columns='month', values='total_amount_sum') #.to_csv('month_amount.csv')
    df_monthly_buy_sum_pivot.columns = [ f'month_{x}_sum_pivot' for x in df_monthly_buy_sum_pivot.columns.ravel()]

    df_monthly_buy_avg_pivot = pd.pivot_table(df_monthly_buy, index='userid', columns='month', values='total_amount_mean') #.to_csv('month_total_amount_mean.csv')
    df_monthly_buy_avg_pivot.columns = [ f'month_{x}_avg_pivot' for x in df_monthly_buy_avg_pivot.columns.ravel()]
    #df_monthly_buy_sum_pivot.head()

    # Calculate the total n purchase days of each user 
    df_purchase_ndays = df_purchase_detail.groupby('userid')['grass_date'].count()

    # Calculate the average purchase amount per order of each user
    df_purchase_detail['amount_per_order'] = df_purchase_detail.total_amount / df_purchase_detail.order_count
    df_purchase_amount_per_order = df_purchase_detail.groupby('userid').agg({'amount_per_order':['sum', 'min', 'max', 'std','mean']})
    df_purchase_amount_per_order.columns = ["_".join(x) for x in df_purchase_amount_per_order.columns.ravel()]

    # Calculate the statistics of the purchase of each user as new features
    df_purchase_detail_feature = df_purchase_detail.groupby('userid').agg({'order_count':['sum', 'min', 'max', 'std', 'mean'],
                                                                        'total_amount':['sum', 'min', 'max', 'std','mean']})
    df_purchase_detail_feature.columns = ["_".join(x) for x in df_purchase_detail_feature.columns.ravel()]

    # Count the total purchase amount of each category for each user
    df_category_count = df_purchase_detail.groupby(['userid','category_encoded'], as_index=False)['total_amount'].sum()
    df_category_pivot = pd.pivot_table(df_category_count, index='userid', 
                                                        columns='category_encoded',
                                                        values='total_amount')
    df_category_pivot = df_category_pivot.fillna(0)
    df_category_pivot.columns = ['category_' + str(x) for x in df_category_pivot.columns]

    # Select the desired features from the dataset created from R code
    df_rfm_features = df_rfm_features[['userid', 'DistinctDay', 'DistinctDayIn90days', 'DistinctDayIn60days',
        'DistinctDayIn30days', 'DistinctDayIn14days', 'DistinctDayIn7days',
        'DistinctDayIn3days', 'FreqIn90days', 'FreqIn60days',
        'FreqIn14days', 'FreqIn3days', 'FreqIn7days', 'rececny']]


    # Apply RFM Model to synethesize the features (WIP)


    ## Integrate all systhesized featurs into both the `df_train` and `df_test` datasets
    # Merge synthesized features back into df_train and df_test
    df_list = [df_login_feature, df_purchase_dt_diff, df_feature_v3, df_feature_v4, df_feature_v5, df_monthly_buy_sum_pivot, df_purchase_ndays, 
            df_purchase_detail_feature, df_purchase_amount_per_order, df_growth_features, df_rfm_features, df_category_pivot] #remove `df_month` and add `df_monthly_buy_sum_pivot`

    # Applied `reduce` function to merge the orginal df_train/df_test and created features
    df_train = reduce(lambda left, right: pd.merge(left, right, on='userid', how='inner'), [df_train, *df_list])
    df_test = reduce(lambda left, right: pd.merge(left, right, on='userid', how='inner'), [df_test, *df_list])


    ## Modeling with Lightgbm
    #df_train.columns #see the columns of `df_train`
    # Split the train and test datasets
    df_tr, df_val = train_test_split(df_train, stratify = df_train['label'], test_size=0.2, random_state=42)

    features = ['userid', 'gender', 'is_seller', 'birth_year', 'enroll_time',
                'age', 'lifetime', 'login_times_sum', 'login_times_min',
                'login_times_max', 'login_times_std', 'login_times_mean', 'dt_diff_sum',
                'dt_diff_min', 'dt_diff_max', 'dt_diff_std', 'dt_diff_mean',
                'AvgMoMOrderCnt', 'AvgMoMTotCnt', 'OrderCntIn90days', 'TotCntIn90days',
                'OrderCntIn60days', 'TotCntIn60days', 'OrderCntIn30days',
                'TotCntIn30days', 'OrderCntIn14days', 'TotCntIn14days',
                'OrderCntIn7days', 'TotCntIn7days', 'OrderCntIn3days', 'TotCntIn3days',
                'BuyRececny', 'DistinctCategory', 'GoodBuyer', 'month_2_sum_pivot',
                'month_3_sum_pivot', 'month_4_sum_pivot', 'month_5_sum_pivot',
                'month_6_sum_pivot', 'month_7_sum_pivot', 'grass_date',
                'order_count_sum', 'order_count_min', 'order_count_max',
                'order_count_std', 'order_count_mean', 'total_amount_sum',
                'total_amount_min', 'total_amount_max', 'total_amount_std',
                'total_amount_mean', 'amount_per_order_sum', 'amount_per_order_min',
                'amount_per_order_max', 'amount_per_order_std', 'amount_per_order_mean',
                'avgFreqMoM', 'DistinctDay', 'DistinctDayIn90days',
                'DistinctDayIn60days', 'DistinctDayIn30days', 'DistinctDayIn14days',
                'DistinctDayIn7days', 'DistinctDayIn3days', 'FreqIn90days',
                'FreqIn60days', 'FreqIn14days', 'FreqIn3days', 'FreqIn7days', 'rececny',
                'category_1', 'category_2', 'category_3', 'category_4', 'category_5',
                'category_6', 'category_7', 'category_8', 'category_9', 'category_10',
                'category_11', 'category_12', 'category_13', 'category_14',
                'category_15', 'category_16', 'category_17', 'category_18',
                'category_19', 'category_20', 'category_21', 'category_22',
                'category_23']
    #print(len(features))

    # Create the training and validation sets
    tr_X = df_tr[features].values
    tr_y = df_tr['label']

    val_X = df_val[features].values
    val_y = df_val['label']

    te_X = df_test[features].values

    # Load the datasets into lgd.Dataset format
    lgtrain = lgb.Dataset(tr_X, tr_y)
    lgvalid = lgb.Dataset(val_X, val_y)

    # Set the training parameters
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

    evals_result = {}

    # Start training
    evals_result = {}

    clf = lgb.train(params,
                    lgtrain,
                    1500,
                    valid_sets = [lgvalid, lgtrain],
                    valid_names = ['validation', 'train'],
    #               feval=lgb_fbeta_score, # We used roc_auc_score this time
                    evals_result = evals_result,
                    early_stopping_rounds = 200,
                    verbose_eval = 50)


    ## Inference and output the result
    # Pedict the labels
    pred_test = clf.predict(te_X)

    # Output the predict labels to `submission.csv`
    sub = pd.read_csv(os.path.join(folder_path, 'submission.csv'))
    sub['label'] = pred_test

    sub.to_csv('submission_final.csv', index=False)