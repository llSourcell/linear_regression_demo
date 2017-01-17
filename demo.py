import pandas as pd;import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
#read data
dataframe = pd.read_csv('challenge_dataset.txt',names = ['Brain','Body'])

print np.size(dataframe,axis=0)
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)
pred = body_reg.predict(x_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, pred)
plt.xlabel('Brain_weights')
plt.ylabel("Body_weights")
plt.show()

print "r-square score:",body_reg.score(x_values,y_values)
print "slope:" ,body_reg.coef_
print "intercept:", body_reg.intercept_
print "MSE value:", mean_squared_error(y_values,pred)
print "RMSE value: ",np.sqrt(mean_squared_error(y_values,pred))


#with train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.33, random_state=1)

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_train, y_train)
pred = body_reg.predict(x_test)

#visualize results
plt.scatter(x_test, y_test)
plt.plot(x_test, pred)
plt.xlabel('Brain_weights')
plt.ylabel("Body_weights")
plt.show()

print "r-square score:",body_reg.score(x_test,y_test)
print "slope:" ,body_reg.coef_
print "intercept:", body_reg.intercept_
print "MSE value:", mean_squared_error(y_test,pred)
print "RMSE value: ",np.sqrt(mean_squared_error(y_test,pred))
