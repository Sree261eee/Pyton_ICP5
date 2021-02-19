"""numeric_features = restaurent_data.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Top 4 correlated variables to the target variable quality is: ')
print(corr['revenue'].sort_values(ascending=False)[1:4],'\n')
print('Top 3 correlated variables to the target variable quality is: ')
print(corr['revenue'].sort_values(ascending=False)[1:3],'\n')"""