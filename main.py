import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


dataframe = pd.read_fwf('brain_body.txt')

y_values = dataframe(['Body'])
x_values = dataframe(['Brain'])

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
