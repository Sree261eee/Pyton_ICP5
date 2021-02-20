import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
restaurent_data = pandas.read_csv("restaurentdata.csv")
train_data = restaurent_data.drop(['revenue','City Group','Type'], axis=1)
print(type(train_data))
test_data = restaurent_data["revenue"].astype(str)
print(type(test_data))
regr = linear_model.LinearRegression()
regr.fit(train_data, test_data)
revenue_pred=regr.predict(train_data)

print("Variance score: %.2f" % r2_score(test_data,revenue_pred))
print("Mean squared error: %.2f" % mean_squared_error(test_data,revenue_pred))  #squaring errors for removing negative values

numeric_features = restaurent_data.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Top 5 correlated variables to the target variable quality is: ')
print(corr['revenue'].sort_values(ascending=False)[0:6],'\n')

quality_pivot = restaurent_data.pivot_table(index=['P2'],values=['revenue'],aggfunc=np.median)
quality_pivot.plot(kind='bar',color ='blue')
plt.show()

corelated_features = restaurent_data[['P2','P28','P6','P21','P11']]
corelated_target = restaurent_data['revenue']
#X_train, X_test,y_train, y_test = train_test_split(corelated_features,corelated_target,test_size=0.33,random_state=42)
regr1 = linear_model.LinearRegression()
regr1.fit(corelated_features,corelated_target)

prediction1=regr1.predict(corelated_features)

print("Variance score: %.2f" % r2_score(corelated_target,prediction1))
print('mse',mean_squared_error(corelated_target,prediction1))

