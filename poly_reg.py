import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("maaslar.csv")

x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)

Xpol=poly_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(Xpol,y)

plt.scatter(x,y, color="red")
plt.plot(x, reg.predict(Xpol), color="blue")
plt.show()

#reg.predict(poly_reg.fit_transform())