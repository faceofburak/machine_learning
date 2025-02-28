import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("maaslar.csv")

x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

from sklearn.tree import DecisionTreeRegressor

r_dt= DecisionTreeRegressor()

r_dt.fit(x,y)

plt.scatter(x,y, color="red")
plt.plot(x,r_dt.predict(x),color="blue")

plt.show()