import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

restaurent_data = pandas.read_csv("restaurentdata.csv")
train_data = restaurent_data.drop(['revenue','City Group','Type'], axis=1)
print(type(train_data))
test_data = restaurent_data["revenue"].astype(str)
print(type(test_data))
regr = linear_model.LinearRegression()
regr.fit(train_data, test_data)
revenue_pred=regr.predict(train_data)

print("Variance score: %.2f" % r2_score(test_data,revenue_pred))
print("Mean squared error: %.2f" % mean_squared_error(test_data,revenue_pred))

