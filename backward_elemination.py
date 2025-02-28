import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('veriler.csv')


ulke = data.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(data.iloc[:,0])



ohe= preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()





c = data.iloc[:,-1:].values




c[:,-1] = le.fit_transform(data.iloc[:,-1])

print(c)
c = ohe.fit_transform(c).toarray()

print(c)

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])

Yas = data.iloc[:,1:4].values

sonuc2 = pd.DataFrame(data=Yas,columns=['boy','kilo','yas'])


sonuc3=pd.DataFrame(data=c[:,:1],columns=['cinsiyet'])

print(sonuc3)

s=pd.concat([sonuc, sonuc2],axis=1)

s2=pd.concat([s, sonuc3],axis=1)

print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(sonuc3, s, test_size=0.33, random_state=0)

boy = s2.iloc[:,3:4].values
sag = s2.iloc[:,4:]
sol = s2.iloc[:,:3]

veri=pd.concat([sol, sag], axis=1)

x_train, x_test, y_train, y_test= train_test_split(veri, boy, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)

print(y_predict)
print(y_test)

import statsmodels.api as sm

#x= np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)
 
X_l= veri.iloc[:,[0,1,2,3,5]].values
X_l= np.array(X_l, dtype=float)

model = sm.OLS(boy,X_l).fit()
print(model.summary())
