
# coding: utf-8
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, 
# algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)[source]
# 读入数据
# 划分数据
name = ['飞行里程','视频游戏占比时间','冰淇淋消耗','分类']
data = pd.read_csv('/Users/lucasie/Desktop/Video/datingTestSet2.txt',sep='\t',header=None,names=name)
data.head()
X = data[name[:-1]]
Y = data[name[-1]]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.8,random_state=28)

# 数据标准化
# # 将特征维度数据标准化,统一量纲
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

# 创建KNN模型
# 训练并测试模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
# y_predict = knn.predict(x_test)
print('训练集成绩:%s'%(knn.score(x_train,y_train)))
print('测试集成绩%s'%(knn.score(x_test,y_test)))
