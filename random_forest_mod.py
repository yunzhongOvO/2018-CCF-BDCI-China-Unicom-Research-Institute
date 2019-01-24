# 引入需要得包
#数据处理常用得包
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# 随机森林的包
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier

# 画图的包
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import warnings
warnings.filterwarnings("ignore")


# 读取数据
# 读取成DataFrame的数据
dt = pd.read_csv(r'D:\telecoms\train_all.csv')
dtest = pd.read_csv(r'D:\telecoms\train_2.csv')



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




# 前600000行的data_ar作为训练数据，之后的作为测试数据来训练模型
data_train = train_data
data_test = test_data

print ("Number of features used for training:\t",len(data_train),
       "\nNumber of features used for testing:\t",len(data_test))



# 开始使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100) # 定义决策树的个数为100

# # 用全部训练数据来做训练
# # target = data_train[:,21].ravel()
# # train = data_train[:,0:21]
model = clf.fit(data_train.astype('int')[:,0:21], data_train.astype('int')[:,21])

# # 用测试集数据来预测最终结果
pred = clf.predict(test_data[:,0:21])
# 计算准确度
acc = np.mean(pred == test_data[:,21].ravel()) *100
print("Accuracy of pure RandomForest classifier"
      ": \t", acc, "%")
# 计算得分
print("F1_score", f1_score(test_data[:,21], pred, average = 'weighted'))











# 1个二值离散变量，int型：service_type，是一个超强的分类特征。可以根据它把数据分为两部分
train_service1 = []
train_service4 = []  # 划分后的数据集

test_service1 = []
test_service4 = []  # 划分后的数据集
#print(len(data_ar))
for i in range(0,len(train.dropna())):
    if train_data[i][0] == 1:
        train_service1.append(train_data[i])
    else:
        train_service4.append(train_data[i])

for i in range(0,len(test.dropna())):
    if test_data[i][0] == 1:
        test_service1.append(test_data[i])
    else:
        test_service4.append(test_data[i])


print ("Number of features used for training(s1):\t",len(train_service1),
       "\nNumber of features used for testing(s1):\t",len(test_service1),
       "\nNumber of features used for training(s4):\t", len(train_service4),
       "\nNumber of features used for testing(s4):\t", len(test_service4),)


# 将DataFrame的数据转换成Array
data_train_s1 = np.array(train_service1)
data_test_s1 = np.array(test_service1)
data_train_s4 = np.array(train_service4)
data_test_s4 = np.array(test_service4)



# 开始使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100) # 定义决策树的个数为100

# # 用全部训练数据来做训练
# # target = data_train[:,21].ravel()
# # train = data_train[:,0:21]
model = clf.fit(data_train_s1.astype('int')[:,0:21], data_train_s1.astype('int')[:,21])

# # 用测试集数据来预测最终结果
pred = clf.predict(data_test_s1[:,0:21])
# 计算准确度
acc = np.mean(pred == data_test_s1[:,21].ravel()) *100
print("Accuracy of pure RandomForest classifier"
      ": \t", acc, "%")
# 计算得分
print("F1_score", f1_score(data_test_s1[:,21], pred, average = 'weighted'))







# 开始使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100) # 定义决策树的个数为100

# # 用全部训练数据来做训练
# # target = data_train[:,21].ravel()
# # train = data_train[:,0:21]
model = clf.fit(data_train_s4.astype('int')[:,0:21], data_train_s4.astype('int')[:,21])

# # 用测试集数据来预测最终结果
pred = clf.predict(data_test_s4[:,0:21])
# 计算准确度
acc = np.mean(pred == data_test_s4[:,21].ravel()) *100
print("Accuracy of pure RandomForest classifier"
      ": \t", acc, "%")
# 计算得分
print("F1_score", f1_score(data_test_s4[:,21], pred, average = 'weighted'))











# 存放不同参数取值，以及对应的精度，每一个元素都是一个三元组(a, b, c)
results = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(1, 500, 3))
# 决策树个数参数取值
n_estimators_options = list(range(1, 1000, 5))
groud_truth = data_test_s4[:,21].ravel()

for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
        alg.fit(data_train_s4.astype('int')[:,0:21], data_train_s4.astype('int')[:,21])
        predict = alg.predict(data_test_s4[:,0:21])
        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
        results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
        # 真实结果和预测结果进行比较，计算准确率
        print((groud_truth == predict).mean())

# 打印精度最大的那一个三元组
print(max(results, key=lambda x: x[2]))