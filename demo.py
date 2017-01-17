import pandas as pd;import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from IPython.display import display
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



'''features = dataframe
poly = PolynomialFeatures(2)
features = poly.fit_transform(dataframe)
features = pd.DataFrame(features,columns =['constant','Brain','Body','Brain**2','Brain*Body','Body**2'])
target = features['Body']
features = features.drop('Body',axis= 1);

lr = LinearRegression()
predicted = cross_val_predict(lr,features,target,cv = 5)
scores = cross_val_score(lr,features,target,cv=5,scoring = 'r2')
scores
fig,ax = plt.subplots()
ax.scatter(target,predicted)
ax.plot([target.min(),target.max()],[target.min(),target.max()],'k--',lw=4)
ax.set_xlabel("Features")
ax.set_ylabel("Body weight")
plt.show()'''
