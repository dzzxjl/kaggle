# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 20:58
# @Author  : dzzxjl@126.com

# -*- coding: utf-8 -*-

# 引入模块
from sklearn.linear_model import LinearRegression
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# 读取数据
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
submit = pd.read_csv("./data/sample_submit.csv")

# 删除id
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# 取出训练集的y
y_train = train.pop('y')

# 建立线性回归模型
reg = LinearRegression()
reg.fit(train, y_train)


feature_importance_list = []
for x, y in zip(train.columns.values, reg.coef_):
    feature_importance_list.append((x, y))
result = sorted(feature_importance_list, key=lambda score: score[1])
# print(result)



# for i in range(7):
#     print(train.columns.values[i], reg.coef_[i])
# print(train.columns.values)
# print(reg.coef_)
y_pred = reg.predict(test)

# print(y_pred)
# print(y_pred)
# 若预测值是负数，则取0
y_pred = map(lambda x: x if x >= 0 else 0, y_pred)

y_pred = map(lambda x: x if x >= 0 else 0, y_pred)

y_pred = list(y_pred)


# print(y_pred)
# 输出预测结果至my_LR_prediction.csv
submit['y'] = y_pred
submit.to_csv('my_LR_prediction.csv', index=False)