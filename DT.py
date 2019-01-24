#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.externals import joblib


# In[2]:


#从原始文件读取：
all_data=pd.read_csv("E://train_all.csv",low_memory=False)#此时的数据为混合格式

#数据处理：去重。结果：没有重复的数据
#np.shape(all_data)
#all_data.drop_duplicates(inplace=True)
#np.shape(all_data)

#数据处理：去除\N。结果：去除共4行。
#all_data.loc[(all_data['2_total_fee'] == '\\N')]#分列检索\N
#all_data.loc[(all_data['age'] == '\\N')]#分列检索\N
#all_data.loc[140569]#可以复查
all_data.drop(140569,inplace=True)#删除
all_data.drop(600438,inplace=True)#删除
all_data.drop(642033,inplace=True)#删除
all_data.drop(232620,inplace=True)#删除
#all_data.shape
#all_data =  all_data.reset_index(drop=True)#重新生成index
#all_data=all_data.dropna()#直接去除\N

#保存数据
all_data.to_csv("E://all_data.csv",index=False)


# In[3]:


all_data=pd.read_csv("E://all_data.csv",low_memory=False)#此时为混合格式
#print(all_data)

#替换current_service
all_data.loc[all_data['current_service']==90063345,'current_service']=1
all_data.loc[all_data['current_service']==89950166,'current_service']=2
all_data.loc[all_data['current_service']==89950167,'current_service']=3
all_data.loc[all_data['current_service']==99999828,'current_service']=4
all_data.loc[all_data['current_service']==90109916,'current_service']=5
all_data.loc[all_data['current_service']==89950168,'current_service']=6
all_data.loc[all_data['current_service']==99999827,'current_service']=7
all_data.loc[all_data['current_service']==99999826,'current_service']=8
all_data.loc[all_data['current_service']==90155946,'current_service']=9
all_data.loc[all_data['current_service']==99999830,'current_service']=10
all_data.loc[all_data['current_service']==99999825,'current_service']=11


# In[4]:


#特征选取
#all_data['service_type'].dtypes#int64

#不论从理解上去分析，还是从特征统计来分析，is_mix_service可以舍弃
#sns.countplot(x="current_service", hue="is_mix_service", data=data)
all_data.drop('is_mix_service',axis=1,inplace=True)#删除
#all_data

#online_time，作为一个连续型的分类依据。没有什么明显特征，待后续处理。
#all_data['online_time'].dtypes#int64

#2个连续float
#total_fee系列，pattern相似
#all_data['1_total_fee'].dtypes#float64
#此处对total_fee系列数据进行基于相关系数的提取与重组
#all_data.iloc[:,2:6]#确认目标数据
#print(all_data.dropna()['1_total_fee'].corr(all_data.dropna()['current_service']))
#print(all_data.dropna()['2_total_fee'].corr(all_data.dropna()['current_service']))
#print(all_data.dropna()['3_total_fee'].corr(all_data.dropna()['current_service']))
#print(all_data.dropna()['4_total_fee'].corr(all_data.dropna()['current_service']))
#print(all_data.iloc[:,2:6].mean(axis=1).corr(all_data.dropna()['current_service']))
#print(all_data.iloc[:,2:6].std(axis=1).corr(all_data.dropna()['current_service']))
#print(all_data.iloc[:,2:6].max(axis=1).corr(all_data.dropna()['current_service']))
#print(all_data.iloc[:,2:6].min(axis=1).corr(all_data.dropna()['current_service']))
#0.5152339722718543
#0.497141808554911
#0.5094022427430654
#0.5000643678789383
#0.5462751177169615
#0.21413992546539737
#0.4703718595573495
#0.5928537502350928
#相关系数只是数学上的概念，需要偏序性，仍然存在很大局限
all_data.insert(6, 'mean_total_fee', all_data.iloc[:,2:6].mean(axis=1))#float64连续型数据
all_data.insert(7, 'min_total_fee', all_data.iloc[:,2:6].min(axis=1))
del all_data['1_total_fee']
del all_data['2_total_fee']
del all_data['3_total_fee']
del all_data['4_total_fee']
#all_data

#3个连续float
#traffic系列，包含month_traffic, last_month_traffic, local_trafffic_month#注意存在拼写错误
#不知道该怎么处理，没有理解，先尽数保留。
#all_data['local_trafffic_month'].dtypes#float64
#all_data[['month_traffic','last_month_traffic','local_trafffic_month','current_service']].corr()

#舍弃：complaint_level，former_complaint_num，former_complaint_fee，net_service
del all_data['complaint_level']
del all_data['former_complaint_num']
del all_data['former_complaint_fee']
del all_data['net_service']

#2个int
#age和gender相关系数不高，但按常理似乎应该保留。
#all_data['gender'].dtypes#int
#all_data[['age','gender','current_service']].corr()

#3个float
#caller_time系列
#all_data['service2_caller_time'].dtype

#缴费
#保留pay_num
#all_data['pay_num'].dtype#float64
#print(all_data['pay_times'].corr(all_data['current_service']))
#print(all_data['pay_num'].corr(all_data['current_service']))
#print(all_data['pay_num'].div(all_data['pay_times']).corr(all_data['current_service']))
del all_data['pay_times']

#合约
#不知道不确定，先保留
#all_data['contract_type'].dtype#int
#all_data['contract_time'].dtype#int

#剩余超额和最低消费
#all_data['many_over_bill'].dtype#int
#all_data['is_promise_low_consume'].dtype#int

print(all_data)#剩余19栏


# In[5]:


#读取测试集（复赛训练集数据）
test=pd.read_csv("E://test.csv",low_memory=False)#此时为混合格式

#格式化
test.loc[test['current_service']==90063345,'current_service']=1
test.loc[test['current_service']==89950166,'current_service']=2
test.loc[test['current_service']==89950167,'current_service']=3
test.loc[test['current_service']==99999828,'current_service']=4
test.loc[test['current_service']==90109916,'current_service']=5
test.loc[test['current_service']==89950168,'current_service']=6
test.loc[test['current_service']==99999827,'current_service']=7
test.loc[test['current_service']==99999826,'current_service']=8
test.loc[test['current_service']==90155946,'current_service']=9
test.loc[test['current_service']==99999830,'current_service']=10
test.loc[test['current_service']==99999825,'current_service']=11
test.drop('is_mix_service',axis=1,inplace=True)
test.insert(6, 'mean_total_fee', test.iloc[:,2:6].mean(axis=1))
test.insert(7, 'min_total_fee', test.iloc[:,2:6].min(axis=1))
del test['1_total_fee']
del test['2_total_fee']
del test['3_total_fee']
del test['4_total_fee']
del test['complaint_level']
del test['former_complaint_num']
del test['former_complaint_fee']
del test['net_service']
del test['pay_times']


# ### 使用决策树框架进行训练，调节决策树深度

# In[6]:


from sklearn import tree
# 训练
X_train = all_data.drop(['user_id','current_service'], axis=1).values.tolist()
y_train = all_data['current_service']
X_test = test.drop(['user_id','current_service'], axis=1).values.tolist()
y_test = test['current_service']
from sklearn.metrics import precision_score, recall_score, f1_score


# In[9]:


depth,score1,score2,score3,score4=[],[],[],[],[]
for i in range(8,24):
    # 初始化决策树分类器
    clf = tree.DecisionTreeClassifier(max_depth = i,
                                     criterion = 'gini',
                                     splitter = 'best',
                                      min_samples_split = 8,                                  
                                     )						#创建DecisionTreeClassifier()类
    clf = clf.fit(X_train, y_train)
    # 预测 保存结果
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    #模型评估
    depth.append(str(i))
    score1.append(clf.score(X_train, y_train))
    score2.append(clf.score(X_test, y_test))
    score3.append(f1_score(y_train, y_train_pred, average='weighted'))
    score4.append(f1_score(y_test, y_test_pred, average='weighted'))
print(depth)
print(score1)
print(score2)
print(score3)
print(score4)


# In[20]:


import matplotlib.pyplot as plt
plt.figure()
plt.plot(depth, score3)
plt.plot(depth, score4 , color='red', linewidth=1.0, linestyle='--')
plt.show()


# In[22]:


df = pd.DataFrame({'depth':depth,'train':score1,'tesst':score2,'f1_train':score3,'f1_test':score4})
print(df)


# ### 使用GridSearchCV对决策树进行调参

# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  cross_val_score
from sklearn.cross_validation import PredefinedSplit


# In[38]:


#设置训练框架，不使用k-fold，使用自定义框架
train_val_features = np.concatenate((X_train,X_test ),axis=0)# 合并训练集和验证集
train_val_labels = np.concatenate((y_train,y_test ),axis=0)
test_fold = np.zeros(train_val_features.shape[0]) # 将所有index初始化为0,0表示第一轮的验证集
test_fold[:len(X_train)] = -1 # 将训练集对应的index设为-1，表示永远不划分到验证集中
ps = PredefinedSplit(test_fold=test_fold)


# 第一轮调参

# In[49]:


param = {'max_depth':[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],'min_samples_split':[4,8,12,16,20,25],'criterion':['gini','entropy']}
grid_search_params = {'estimator': DecisionTreeClassifier(),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[86]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
criterion,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    criterion.append(cv_result['parameters'][i]['criterion'])
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])
    score.append(cv_result['cv_validation_scores'][i])
df = pd.DataFrame({'criterion':criterion,
                   'max_depth':max_depth,
                   'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[81]:


with open('E://cv_result1.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[90]:


#可视化作图与分析
import seaborn as sns
sns.pointplot(x="max_depth", y="score",
#              hue="min_samples_split" ,
              data=df)


# 第一轮条调参结束，开始第二轮调参

# In[98]:


param = {'max_depth':[13,14,15,16,17],'min_samples_split':[20,25,30,35,40],'min_samples_leaf':[1,2,4],'criterion':['gini'],}
grid_search_params = {'estimator': DecisionTreeClassifier(),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[110]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[109]:


with open('E://cv_result2.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[113]:


import seaborn as sns
sns.pointplot(x="max_depth", y="score",
              hue="min_samples_leaf" ,
              data=df)


# In[118]:


param = {'max_depth':[15,16,17],'min_samples_split':[30,50,70,90,110],'min_samples_leaf':[2,4,8,16,32,64]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[121]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[120]:


with open('E://cv_result3.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[126]:


import seaborn as sns
sns.pointplot(x="max_depth", y="score",
#              hue="min_samples_split" ,
              data=df)


# In[127]:


param = {'max_depth':[16,17,18],'min_samples_split':[45,50,55,60,65],'min_samples_leaf':[12,14,16,18,20]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[132]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[131]:


with open('E://cv_result4.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[135]:


import seaborn as sns
sns.pointplot(x="max_depth", y="score",
#              hue="min_samples_split" ,
              data=df)


# In[136]:


sns.pointplot(x="max_depth", y="score",
              hue="min_samples_leaf" ,
              data=df)


# In[137]:


sns.pointplot(x="max_depth", y="score",
              hue="min_samples_split" ,
              data=df)


# In[138]:


param = {'max_depth':[18,20,22],'min_samples_split':[55,65,75],'min_samples_leaf':[15,17,19]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[144]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[140]:


with open('E://cv_result5.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[149]:


sns.pointplot(hue="max_depth", y="score",
              x="min_samples_split" ,
              data=df)


# In[153]:


param = {'max_depth':[22,24,26],'min_samples_split':[60,65,70],'min_samples_leaf':[15,17,19]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[157]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[155]:


with open('E://cv_result6.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[159]:


sns.pointplot(x="max_depth", y="score",
              hue="min_samples_leaf" ,
              data=df)


# In[162]:


param = {'max_depth':[22,24,26],'min_samples_split':[61,63,65,67],'min_samples_leaf':[15,16,17,18]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[165]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[164]:


with open('E://cv_result7.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[166]:


sns.pointplot(x="max_depth", y="score",
              hue="min_samples_split" ,
              data=df)


# In[167]:


param = {'max_depth':[20,22,24],'min_samples_split':[63,65,67],'min_samples_leaf':[16,18,20]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[170]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[169]:


with open('E://cv_result9.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[172]:


sns.pointplot(x="max_depth", y="score",
              hue="min_samples_leaf" ,
              data=df)


# In[173]:


param = {'max_depth':[21,22,23],'min_samples_split':[65,66,67],'min_samples_leaf':[17,18,19]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[177]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
#    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[175]:


with open('E://cv_result10.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# In[178]:


sns.pointplot(x="max_depth", y="score",
              hue="min_samples_split" ,
              data=df)


# In[179]:


param = {'max_depth':[21,22,23],'min_samples_split':[63,64,65,66],'min_samples_leaf':[17,18,19]}
grid_search_params = {'estimator': DecisionTreeClassifier(criterion='gini'),
'param_grid': param, # 前面定义的我们想要优化的参数
'cv': ps, # 使用前面自定义的split验证策略
'n_jobs': -1} # 并行运行的任务数，-1表示使用所有CPU
gridsearch = GridSearchCV(**grid_search_params)
gridsearch.fit(train_val_features, train_val_labels)


# In[180]:


import pandas as pd
cv_result = pd.DataFrame.from_dict(gridsearch.grid_scores_)
min_samples_leaf,max_depth,min_samples_split,score=[],[],[],[]
for i in range(len(cv_result)):
    max_depth.append(cv_result['parameters'][i]['max_depth'])
    min_samples_split.append(cv_result['parameters'][i]['min_samples_split'])
    min_samples_leaf.append(cv_result['parameters'][i]['min_samples_leaf'])
    score.append(str(cv_result['cv_validation_scores'][i]).split('[')[1].split(']')[0])#做表格要用到str
#    score.append(cv_result['cv_validation_scores'][i])#作图要用到int
df = pd.DataFrame({
                   'max_depth':max_depth,
                   'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split,
                   'score':score
                  })


# In[181]:


with open('E://cv_result11.csv','w') as f:
    df.to_csv(f)
print('The parameters of the best model are: ')
print(gridsearch.best_params_)


# 调参结束

# In[182]:


y_pred = gridsearch.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=y_pred))

