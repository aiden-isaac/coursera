import numpy as np
import pandas as pd
import matplotlib.pylab as plt

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"

df = pd.read_csv(file_path, header=0)

df['Screen_Size_cm'] = np.round(df[['Screen_Size_cm']], 2)

missing_data = df.isnull()

avg_weight = df['Weight_kg'].astype("float").mean(axis=0)
df['Weight_kg'].replace(np.nan, avg_weight)

freq = df['Screen_Size_cm'].value_counts().idxmax()
df['Screen_Size_cm'].replace(np.nan, freq)

df['Weight_kg'] = df['Weight_kg'].astype("float")
df['Screen_Size_cm'] = df['Screen_Size_cm'].astype("float")

df['Screen_Size_cm'] = df['Screen_Size_cm']/2.54
df = df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'})

df['Weight_kg'] = df['Weight_kg']*2.205
df = df.rename(columns={'Weight_kg':'Weight_lb'})

df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()

groups = ["Low", "Medium", "High"]

bins = np.linspace(df['Price'].min(), df['Price'].max(), 4)

df['Price-binned'] = pd.cut(df['Price'], bins, labels=groups, include_lowest=True)

plt.bar(groups, df['Price-binned'].value_counts())

plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price bins")

dummy = pd.get_dummies(df['Screen'])

df['Screen'] = df['Screen'].rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'})

df = pd.concat([df, dummy], axis=1)

df = df.drop('Screen', axis=1)