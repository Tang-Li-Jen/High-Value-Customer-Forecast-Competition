import pandas as pd
import numpy as np


purchase_detail = pd.read_csv('purchase_detail.csv')
purchase_detail.grass_date = pd.to_datetime(purchase_detail.grass_date)
purchase_detail['month'] = purchase_detail.grass_date.dt.month

data = purchase_detail.groupby(['userid', 'month']).agg({'order_count':['sum', 'mean'], 'total_amount':['sum', 'mean']})
data.columns = ["_".join(x) for x in data.columns.ravel()]
pd.pivot_table(data, index='userid', columns='month', values='total_amount_sum').to_csv('month_amount.csv')
pd.pivot_table(data, index='userid', columns='month', values='total_amount_mean').to_csv('month_total_amount_mean.csv')