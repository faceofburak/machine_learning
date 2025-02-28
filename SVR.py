import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("maaslar.csv")

x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_scaled=sc.fit_transform(x)
sc2=StandardScaler()
y_scaled=sc.fit_transform(y)

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")

svr_reg.fit(x_scaled,y_scaled)

prediction = svr_reg.predict(x_scaled)

plt.scatter(x_scaled,y_scaled, color="red")
plt.plot(x_scaled,svr_reg.predict(x_scaled), color="blue")
plt.show()