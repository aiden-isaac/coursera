import pandas as pd
import numpy as np

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

# List of headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# Reading the data from the URL
df = pd.read_csv(url, header=None)

# Assigning headers to the dataframe
df.columns = headers

# Replacing "?" with NaN
df1 = df.replace("?", np.nan)

# Dropping missing values along the column "price"
df = df1.dropna(subset=["price"], axis=0)

# n being number of top lines (tail(n) for bottom)
n = 5
print(df.head(n))

# Saving the dataframe to a CSV file
#df.to_csv("automobile.csv", index=False)

# Checking the data types of the columns
print("\n" + str(df.dtypes))

# Describing the data
print("\n" + str(df.describe(include="all")))

# Describing the data with specific columns
print("\n" + str(df[["length", "compression-ratio"]].describe()))

# Describing the data with specific columns
print("\n" + str(df.info))