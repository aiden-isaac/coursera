import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df = pd.read_csv(file_path, header=0)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)

# Print the correlation between the following columns: bore, stroke, 
# compression-ratio, and horsepower. A lower value means a weak 
# relationship, while a higher value means a strong relationship.
#print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Engine size as potential predictor variable of price
# Plot a scatter plot of 'engine-size' and 'price'
sns.regplot(x='engine-size', y='price', data=df)
plt.ylim(0,)
#plt.savefig(fname='engine-size_vs_price.png')
plt.clf()  # Clear the current figure

# Find the correlation between highway-mpg and price
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#plt.savefig(fname='highway-mpg_vs_price.png')
plt.clf()  # Clear the current figure

# Find the correlation between stroke and price
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#plt.savefig(fname='peak-rpm_vs_price.png')
plt.clf()  # Clear the current figure

# Find the correlation between stroke and price
sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
#plt.savefig(fname='stroke_vs_price.png')
plt.clf()  # Clear the current figure

# Find the correlation between body-style and price
sns.boxplot(x="body-style", y="price", data=df)
#plt.savefig(fname='body-style_vs_price.png')
plt.clf()  # Clear the current figure

# Find the correlation between engine-location and price
sns.boxplot(x="engine-location", y="price", data=df)
#plt.savefig(fname='engine-location_vs_price.png')
plt.clf()  # Clear the current figure

# Find the correlation between drive-wheels and price
sns.boxplot(x="drive-wheels", y="price", data=df)
#plt.savefig(fname='drive-wheels_vs_price.png')
plt.clf() # Clear the current figure

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.reset_index(inplace=True)
drive_wheels_counts = drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'})
drive_wheels_counts.index.name = 'drive-wheels'
#print(drive_wheels_counts)

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.reset_index(inplace=True)
engine_loc_counts = engine_loc_counts.rename(columns={'engine-location': 'value_counts'})
engine_loc_counts.index.name = 'engine-location'
#print(engine_loc_counts)

# grouping results
df_group_one = df[['drive-wheels', 'body-style', 'price']]
df_grouped = df_group_one.groupby(['drive-wheels'], as_index=False).agg({'price': 'mean'})
#print(df_grouped)

# grouping results
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
#print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
#print(grouped_pivot)

# Use the "groupby" function to find the average "price" of each car based on "body-style"
df_group_two = df[['body-style', 'price']]
df_group_two = df_group_two.groupby(['body-style'], as_index=False).mean()
#print(df_group_two)

# Heatmap of the pivot table
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
#plt.savefig(fname='heatmap.png')
plt.clf()  # Clear the current figure

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a pseudocolor plot with a specified colormap
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# Label names for the columns and rows
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# Move ticks and labels to the center of the cells
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# Insert labels for the ticks
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# Rotate x-axis labels if they are too long
plt.xticks(rotation=90)

# Add a colorbar to the plot
fig.colorbar(im)

# Save the figure as a PNG file
#plt.savefig(fname='heatmap.png')

# Show the Pearson Correlation Coefficient and P-value of the following columns:
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
pearson_coef, p_value = stats.pearsonr(df['compression-ratio'], df['price'])
#print ("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
