import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

filepath="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"

df = pd.read_csv(filepath, header=0)

sns.regplot(x="CPU_frequency", y="Price", data=df)
plt.ylim(0,)
#plt.savefig(fname="bruh")
plt.clf()

sns.regplot(x="Screen_Size_inch", y="Price", data=df)
plt.ylim(0,)
#plt.savefig(fname="bruh2")
plt.clf()

sns.regplot(x="Weight_pounds", y="Price", data=df)
plt.ylim(0,)
#plt.savefig(fname="bruh3")
plt.clf()

pearson_coef, p_value = stats.pearsonr(df["CPU_frequency"], df['Price'])
#print(str(pearson_coef))
pearson_coef, p_value = stats.pearsonr(df["Screen_Size_inch"], df['Price'])
#print(str(pearson_coef))
pearson_coef, p_value = stats.pearsonr(df["Weight_pounds"], df['Price'])
#print(str(pearson_coef))

sns.boxplot(x="Category", y="Price", data=df)
#plt.savefig(fname="test")

group = df[["GPU", "CPU_core", "Price"]]
grouped = group.groupby(["GPU", "CPU_core"], as_index=False).mean()
pivot = grouped.pivot(index="GPU", columns="CPU_core")

fig, ax = plt.subplots()
im = ax.pcolor(pivot, cmap='RdBu')

#label names
row_labels = pivot.columns.levels[1]
col_labels = pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)

for param in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']:
    pearson_coef, p_value = stats.pearsonr(df[param], df['Price'])
    print(param)
    print("The Pearson Correlation Coefficient for ",param," is", pearson_coef, " with a P-value of P =", p_value)