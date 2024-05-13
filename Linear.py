import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])
print('This are values in x:',x)
print('This are values in y:',y)
Ex4_1=LinearRegression()
print(Ex4_1)
Ex4_1.fit(x,y)
r_sq = Ex4_1.score(x,y)
print(r_sq)
intercept=Ex4_1.intercept_
gradient=Ex4_1.coef_
print('The y-intercept is', intercept)
print('The gradient is',gradient)
y_pred=Ex4_1.predict(x)
print(y_pred)
x_new=np.arange(5).reshape((-1,1))
print(x_new)
y_new=Ex4_1.predict(x_new)
print(y_new)




