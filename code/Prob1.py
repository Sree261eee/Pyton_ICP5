import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

read_dataset = pd.read_csv('data.csv')

garage_area = read_dataset['GarageArea']
sales_price = read_dataset['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')
plt.show()

data_all = pd.concat([read_dataset['GarageArea'], read_dataset['SalePrice']], axis=1)
z = np.abs(stats.zscore(data_all))
threshold = 3
data = data_all[(z < 3).all(axis=1)]
data_anom = data_all[(z >= 3).all(axis=1)]

# Scatter Plot after removing outlier
garage_area = data['GarageArea']
sales_price = data['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')
plt.show()