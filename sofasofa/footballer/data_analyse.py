# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 18:00
# @Author  : dzzxjl@126.com


import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
submit = pd.read_csv("./data/sample_submit.csv")


# print(train.columns)

y_train = train.pop('y')

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
train.drop(['birth_date', 'work_rate_att', 'work_rate_def'], axis=1, inplace=True)
test.drop(['birth_date', 'work_rate_att', 'work_rate_def'], axis=1, inplace=True)

# print(train.isnull())
# print(train.describe())
print(train.info())
#
# reg = Ridge(alpha=.5)
# reg.fit(train, y_train)
# print(reg.coef_)

# 线性回归模型
# reg = LinearRegression()
# reg.fit(train, y_train)

# print(reg.coef_)

# 建立一个默认的xgboost回归模型
# reg = xgb.XGBRegressor()
# reg.fit(train, y_train)
# y_pred = reg.predict(test)

# 输出预测结果至my_XGB_prediction.csv
# submit['y'] = y_pred
# submit.to_csv('my_XGB_prediction.csv', index=False)

