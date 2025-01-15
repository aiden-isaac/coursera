# imports
import pandas as pd # managing the data
import numpy as np  # mathematical operations

import seaborn as sns # visualizing the data
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler # machine learning functions
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm # stastical computations
from scipy import stats

file = 'data/Ames_Housing_Data1.tsv'

house = pd.read_csv(file, sep='\t')

print(house.head(5))

# get more info

print(house.info())

# describe() gets count, mean, min, max, etc.

print(house['SalePrice'].describe())

# for object types

print(house['Sale Condition'].value_counts())

# looking for correlations

hous_num = house.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] # -1 means that the latest row is SalePrice
top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False) #displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

# create a pairplot to check the correlations

#for i in range(0, len(hous_num.columns), 5):
#    sns.pairplot(data = hous_num,
#                x_vars=hous_num.columns[i:i+5],
#                y_vars=['SalePrice'])
#plt.show()

# most correlated is overall qual, Gr Liv Area, Garage Cars, Garage Area and others

# log transformations

print("Skewness: %f" % house['SalePrice'].skew())
sp_untransformed = sns.displot(house['SalePrice'])
plt.show()

# np.log() can tranform so it is more distributed

plt.clf()
log_transformed = np.log(house['SalePrice'])
sp_transformed = sns.displot(log_transformed)
plt.show()
print("Skewness: %f" % (log_transformed).skew())

# Handling duplicates

duplicate = house[house.duplicated(['PID'])]
dup_removed = house.drop_duplicates()
# house.drop_duplicates(subset=['Order'])

# can check for duplicate indexes

house.index.is_unique

# handling missing values

plt.clf()

# expose the null

total = house.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)
plt.show()

# You can drop the missing values

house.dropna(subset=["Lot Frontage"])

# You can drop the entire column

house.drop('Lot Frontage', axis = 1)

# you can then replace the missing values with median values

median = house["Lot Frontage"].median()
house["Lot Frontage"].fillna(median, inplace=True)

# feature scaling

# normalize

norm_data = MinMaxScaler().fit_transform(hous_num)

# standardize

scaled_data = StandardScaler().fit_transform(hous_num)

# Handling outliers

# find them with the box plots

plt.clf()
sns.boxplot(x=house['Lot Area'])
plt.show()

plt.clf()
sns.boxplot(['SalePrice'])
plt.show()

# Bi-variate analysis

# scatterplot

plt.clf()
price_area = house.plot.scatter(x='Gr Liv Area', y = 'SalePrice')
plt.show()

# delete the outliers

house.sort_values(by = 'Gr Liv Area', ascending=False)[:2]
outliers_dropped = house.drop(house.index[[1499,2181]])
new_plot = outliers_dropped.plot.scatter(x = 'Gr Liv Area', y = 'SalePrice')

plt.clf()

# z-score to find outliers mathematically
# if z-score threshold (-3 - 3) outliers

house['LQFSF_Stats'] = stats.zscore(house['Low Qual Fin SF'])
print(house[['Low Qual Fin SF', 'LQFSF_Stats']].describe().round(3))