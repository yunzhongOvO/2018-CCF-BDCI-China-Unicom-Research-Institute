import  numpy as np
import pandas as pd

# 我们将载入seaborn,但是因为载入时会有警告出现，因此先载入warnings，忽略警告
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt

from pandas import DataFrame,Series
dt = pd.read_csv(r'D:\telecoms\train_all.csv')

#去除空值
data = dt.dropna()

#打印数据信息
data.info()

#转换数据类型,将object型转换为可运算的数据类型
data = data.convert_objects(convert_numeric=True)
#再次查看数据信息
data.info()

#是否有缺失数据
#print(data.isnull().any())


#替换current_service
data.ix[data['current_service']==90063345,'current_service']=1
data.ix[data['current_service']==89950166,'current_service']=2
data.ix[data['current_service']==89950167,'current_service']=3
data.ix[data['current_service']==99999828,'current_service']=4
data.ix[data['current_service']==90109916,'current_service']=5
data.ix[data['current_service']==89950168,'current_service']=6
data.ix[data['current_service']==99999827,'current_service']=7
data.ix[data['current_service']==99999826,'current_service']=8
data.ix[data['current_service']==90155946,'current_service']=9
data.ix[data['current_service']==99999830,'current_service']=10
data.ix[data['current_service']==99999825,'current_service']=11

#current_service共有11种，每种有多少个人
print(data["current_service"].value_counts())

#箱线图
sns.boxplot(x = data['current_service'],y = data['former_complaint_fee'])
plt.show()

#两者相关系数
print(data.dropna()['former_complaint_fee'].corr(data.dropna()['current_service']))

#两者关系图
sns.pairplot(data.dropna(), vars=["current_service", "former_complaint_fee"],hue=None,palette="husl")
plt.show()

#将current_service这一列作为索引
#data.index = data['current_service'].tolist()


#xs=data['current_service']
#ys=data['service_type']
#plt.scatter(xs,ys)
#plt.show()


#热点图
data = data.corr()
sns.heatmap(data)
plt.show()















#print(data["service_type"].value_counts())
#print(data["current_service"].value_counts())
#sns.pairplot(data, vars=['current_service','month_traffic'])

#sns.distplot(data['month_traffic'], kde=False)
#g = sns.pairplot(data)
#sns.plt.show()

