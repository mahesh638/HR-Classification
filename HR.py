import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

dataset= pd.read_csv('HR.csv')
X=dataset.iloc[:,1:13]
y=dataset.iloc[:,-1]
m= np.shape(X)[0]
n= np.shape(X)[1]

#Age bin
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
est.fit(X.iloc[:,6:7])
pp=est.transform(X.iloc[:,6:7])
xx=pd.get_dummies(pp.flatten())

#Legnth of service bin
est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
est.fit(X.iloc[:,8:9])
pp=est.transform(X.iloc[:,8:9])
xx1=pd.get_dummies(pp.flatten())

#Avg training score bin
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
est.fit(X.iloc[:,11:12])
pp=est.transform(X.iloc[:,11:12])
xx2=pd.get_dummies(pp.flatten())

X=X.drop(columns=["age"])

X=pd.concat([X, xx], axis=1)


categorical=[]
for i in range(0,n):
  if X.iloc[:,i].dtype.name == 'object':
    categorical.append(i)

from sklearn.preprocessing import Imputer    
for i in range(0,n):
  if i not in categorical:
    imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
    imputer = imputer.fit(X.iloc[:, i:i+1])
    X.iloc[:, i:i+1] = imputer.transform(X.iloc[:, i:i+1])

arr2=np.ones((m,1))
for i in categorical:
  arr1= pd.get_dummies(X.iloc[:,i]).iloc[:,1:].to_numpy()
  arr2=np.append(arr2, arr1, axis=1)
  
arr2=np.delete(arr2, 0, axis=1)
X=X.to_numpy()
X=np.delete(X, categorical, axis=1)
X=np.append(X,arr2, axis=1)

    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
ev = pca.explained_variance_ratio_

from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
X_test, y_test = sm.fit_resample(X_test, y_test)





from xgboost import XGBClassifier
clas=XGBClassifier()
clas.fit(X_train, y_train)
predx=clas.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predx)
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, predx)


#Submission
Test= pd.read_csv('test_2umaH9m.csv')
X1=Test.iloc[:,1:14]
m= np.shape(X1)[0]
n= np.shape(X1)[1]


from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
est.fit(X1.iloc[:,6:7])
pp=est.transform(X1.iloc[:,6:7])
xx=pd.get_dummies(pp.flatten())



categorical=[]
for i in range(0,n):
  if X1.iloc[:,i].dtype.name == 'object':
    categorical.append(i)

from sklearn.preprocessing import Imputer    
for i in range(0,n):
  if i not in categorical:
    imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
    imputer = imputer.fit(X1.iloc[:, i:i+1])
    X1.iloc[:, i:i+1] = imputer.transform(X1.iloc[:, i:i+1])

arr2=np.ones((m,1))
for i in categorical:
  arr1= pd.get_dummies(X1.iloc[:,i]).iloc[:,1:].to_numpy()
  arr2=np.append(arr2, arr1, axis=1)
  
arr2=np.delete(arr2, 0, axis=1)
X1=X1.to_numpy()
X1=np.delete(X1, categorical, axis=1)
X1=np.append(arr2,X1, axis=1)

X1=np.append(xx,X1,axis=1)


acpred= classifier.predict(X1)
acpred = (acpred > 0.5)
eid= np.reshape(Test.iloc[:,0].to_numpy(), (-1,1))
Submit=np.append(eid, acpred, axis=1)
np.savetxt("foo.csv", Submit, delimiter=",")
