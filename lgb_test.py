# 引入需要的包
#数据处理常用的包
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV  # Perforing grid search

# 随机森林的包
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier

# 画图的包
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import warnings
warnings.filterwarnings("ignore")


#%%
# 读取数据
# 读取成DataFrame的数据
dt = pd.read_csv(r'C:\Users\25535\python_work\ML_program\data\train_all\train_all.csv')
dtest = pd.read_csv(r'C:\Users\25535\python_work\ML_program\data\train_2.csv')

# 数据处理------------------------

# 去除空值
train = dt.dropna()
train = train.convert_objects(convert_numeric=True)
test = dtest.dropna()
test = test.convert_objects(convert_numeric=True)

# 替换current_service
train.ix[train['current_service'] == 90063345, 'current_service'] = 1
train.ix[train['current_service'] == 89950166, 'current_service'] = 2
train.ix[train['current_service'] == 89950167, 'current_service'] = 3
train.ix[train['current_service'] == 99999828, 'current_service'] = 4
train.ix[train['current_service'] == 90109916, 'current_service'] = 5
train.ix[train['current_service'] == 89950168, 'current_service'] = 6
train.ix[train['current_service'] == 99999827, 'current_service'] = 7
train.ix[train['current_service'] == 99999826, 'current_service'] = 8
train.ix[train['current_service'] == 90155946, 'current_service'] = 9
train.ix[train['current_service'] == 99999830, 'current_service'] = 10
train.ix[train['current_service'] == 99999825, 'current_service'] = 11

test.ix[test['current_service'] == 90063345, 'current_service'] = 1
test.ix[test['current_service'] == 89950166, 'current_service'] = 2
test.ix[test['current_service'] == 89950167, 'current_service'] = 3
test.ix[test['current_service'] == 99999828, 'current_service'] = 4
test.ix[test['current_service'] == 90109916, 'current_service'] = 5
test.ix[test['current_service'] == 89950168, 'current_service'] = 6
test.ix[test['current_service'] == 99999827, 'current_service'] = 7
test.ix[test['current_service'] == 99999826, 'current_service'] = 8
test.ix[test['current_service'] == 90155946, 'current_service'] = 9
test.ix[test['current_service'] == 99999830, 'current_service'] = 10
test.ix[test['current_service'] == 99999825, 'current_service'] = 11

# # 删除test里current_service的异常值999999
# for i in test['current_service']:
#     if(test['current_service'][i] == 999999):
#         del test[i, :]

# 删除is_mix_service这一特征
train.drop('is_mix_service',axis=1,inplace=True)
test.drop('is_mix_service',axis=1,inplace=True)

#4个连续float型，total_fee系列（这四个模式相似的特征可以整合重组，形成新的特征）
train.insert(6, 'total_fee_mean', train.iloc[:,2:6].mean(axis=1))   #float64连续型数据
test.insert(6, 'total_fee_mean', test.iloc[:,2:6].mean(axis=1))     #float64连续型数据
train.insert(7, 'total_fee_min', train.iloc[:,2:6].min(axis=1))
test.insert(7, 'total_fee_min', test.iloc[:,2:6].min(axis=1))
del train['1_total_fee']
del train['2_total_fee']
del train['3_total_fee']
del train['4_total_fee']
del test['1_total_fee']
del test['2_total_fee']
del test['3_total_fee']
del test['4_total_fee']

#3个连续float型
#traffic系列，包含month_traffic, last_month_traffic, local_trafffic_month#注意存在拼写错误
train.insert(5, 'traffic_month_sum', train.iloc[:,4]+train.iloc[:,13])  #float64连续型数据
test.insert(5, 'traffic_month_sum', test.iloc[:,4]+test.iloc[:,13])     #float64连续型数据

traffic_month_max = []
for i in range(train.shape[0]):
    if train['month_traffic'][i]>train['last_month_traffic'][i]:
        traffic_month_max.append(train['month_traffic'][i])
    else:
        traffic_month_max.append(train['last_month_traffic'][i])
train.insert(6,column='traffic_month_max',value=traffic_month_max)
del train['month_traffic']
del train['last_month_traffic']

traffic_month_max_test = []
for i in range(test.shape[0]):
    if test['month_traffic'][i]>test['last_month_traffic'][i]:
        traffic_month_max_test.append(test['month_traffic'][i])
    else:
        traffic_month_max_test.append(test['last_month_traffic'][i])
test.insert(6,column='traffic_month_max',value=traffic_month_max_test)
del test['month_traffic']
del test['last_month_traffic']

#3个float型，caller_time系列
#all_data['service2_caller_time'].dtype
train.insert(17, 'call_time_max', train.iloc[:,15:17].max(axis=1))
train.insert(18, 'call_time_min', train.iloc[:,15:17].min(axis=1))
train.insert(19, 'call_time_local', train.iloc[:,18]+train.iloc[:,14])
test.insert(17, 'call_time_max', test.iloc[:,15:17].max(axis=1))
test.insert(18, 'call_time_min', test.iloc[:,15:17].min(axis=1))
test.insert(19, 'call_time_local', test.iloc[:,18]+test.iloc[:,14])

#缴费
#保留pay_num
train.insert(13, 'pay_num_pertimes', train.iloc[:,11]/train.iloc[:,12])
test.insert(13, 'pay_num_pertimes', test.iloc[:,11]/test.iloc[:,12])
del train['pay_times']
del test['pay_times']

#舍弃：complaint_level，former_complaint_num，former_complaint_fee，net_service（相关性很低）
del train['complaint_level']
del train['former_complaint_num']
del train['former_complaint_fee']
del train['net_service']
del test['complaint_level']
del test['former_complaint_num']
del test['former_complaint_fee']
del test['net_service']

#2个int型：age和gender，相关系数不高，但按常理似乎应该保留。
test['age'] = [float(test['age'][i]) for i in test['age']]
test['gender'] = [float(test['gender'][i]) for i in test['gender']]
# for i in :
#     data[i] = data[i].astype(float)

# user_id删除
del train['user_id']
del test['user_id']

print(train.info())
print(test.info())


# 将DataFrame的数据转换成Array
train_data = (train.dropna()).values
test_data = (test.dropna()).values

# lgb model---------

# lgb_params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'silent':0,
#     'learning_rate': 0.05,
#     'num_leaves': 50,          # should < 2^(max_depth)
#     'max_depth': -1,           # no limit
#     'min_child_samples': 15,   # Minimum number of data need in a child(min_data_in_leaf)
#     'max_bin': 200,            # Number of bucketed bin for feature values
#     'subsample': 0.8,          # Subsample ratio of the training instance.
#     'subsample_freq': 1,       # frequence of subsample, <=0 means no enable
#     'colsample_bytree': 0.5,   # Subsample ratio of columns when constructing each tree.
#     'min_child_weight': 0,     # Minimum sum of instance weight(hessian) needed in a child(leaf)
#     'subsample_for_bin': 200000,  # Number of samples for constructing bin
#     'min_split_gain': 0,       # lambda_l1, lambda_l2 and min_gain_to_split to regularization
#     'reg_alpha': 2.0,         # L1 regularization term on weights
#     'reg_lambda': 1.0,         # L2 regularization term on weights
#     'verbose': 0,
# }

# num_round = 10
# bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
# num_round = 10
# lgb.cv(param, train_data, num_round, nfold = 5)
# bst = lgb.train(param, train_data, num_round, valid_sets = valid_sets, early_stopping_rounds = 10)
# bst.save_model('model.txt', num_iteration=bst.best_iteration)

clf = lgb.LGBMClassifier(
                objective = 'multiclass',
                boosting_type = 'gbdt',
                num_class = 15,
                lambda_l1 = 0.1,        #权重的L1正则化项
                lambda_l2 = 0.1,        #权重的L2正则化项
                num_leaves = 50,
                max_depth = 6,          #树的最大深度。这个值是用来控制过拟合的
                learning_rate = 0.1,    #每一步迭代步长
                random_state = 2018,
                colsample_bytree = 0.8, #每棵随机采样的列数的占比(每一列是一个特征)
                subsample = 0.9,        #每棵树随机采样的比例
                n_estimators = 380,     #迭代次数
                n_jobs = 10,
                silent = True
)

# x_train, x_test, y_train, y_test = train_test_split(data_train, data_target, test_size = 0.3, shuffle = True, random_state = 2018)
model = clf.fit(
            train_data.astype('int')[:,0:21], 
            train_data.astype('int')[:,21],
            eval_set = [(test_data[:,0:21], test_data[:,21])],
            verbose = 1
         )
pred = clf.predict(test_data[:,0:21])
acc = np.mean(pred == test_data[:,21].ravel()) *100
print("Accuracy of LGB Classifier: \t", acc, "%")
print("F1_score", f1_score(test_data[:,21], pred, average = 'macro'))




