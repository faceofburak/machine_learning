import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('eksikveriler.csv')


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = data.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])

ulke = data.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(data.iloc[:,0])



ohe= preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])

sonuc2 = pd.DataFrame(data=Yas,columns=['boy','kilo','yas'])


cinsiyet = data.iloc[:,4].values

sonuc3=pd.DataFrame(data=cinsiyet,columns=['cinsiyet'])


s=pd.concat([sonuc, sonuc2],axis=1)

s2=pd.concat([s, sonuc3],axis=1)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(sonuc3, s, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

print(Y_train,y_train)