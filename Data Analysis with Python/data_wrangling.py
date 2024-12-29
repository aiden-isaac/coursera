import pandas as pd
import matplotlib.pylab as plt
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style", "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type", "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower", "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, names=headers)

df.replace("?", np.nan, inplace = True)

print(df.head(5))

# Evaluating for Missing Data
missing_data = df.isnull()

# Count missing values in each column (True = missing value, False = not missing value)
print(missing_data.head(5))

# Count missing values in each column
for column in missing_data.columns.values.tolist():
    
    # Print column name
    print(column)

    # Print number of missing values
    print(missing_data[column].value_counts())

    # Print a blank line
    print("")

# Deal with missing data

#for column in missing_data.columns.values.tolist():
#    average = df[column].astype("float").mean(axis=0)
#    print("Average of", column, ":", average)
#    df[column].replace(np.nan, average, inplace=True)

# Normalzed-losses
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Bore
avg_bore = df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)

# Stroke
avg_stroke = df.stroke.astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace = True)

# Num of doors counting and finding the most common value
df["num-of-doors"].value_counts()
df["num-of-doors"].value_counts().idxmax()

# Replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# Drop all rows that do not have price data
df.dropna(subset=["price"], axis=0, inplace=True)

# Reset index, because we dropped two rows
df.reset_index(drop=True, inplace=True)

print(df.head(5))

# Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print(df.dtypes)

# Data Standardization

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["city-L/100km"] = 235/df["city-mpg"]

# Check your transformed data
print(df.head(5))

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# Rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={"highway-mpg":"highway-L/100km"}, inplace=True)

# Check your transformed data
print(df.head(5))

# Data Normalization

# Replace (original value) by (original value)/(maximum value)
df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/df["width"].max()

df["height"] = df["height"]/df["height"].max()

# Binning

# Convert data to correct format
# Fix for FutureWarning: Avoid using inplace=True with chained assignment
df["horsepower"] = df["horsepower"].replace(np.nan, 0)
df["horsepower"] = df["horsepower"].astype(int, copy=True)

# Plotting the histogram of horsepower to see its distribution
plt.hist(df["horsepower"])

# Set x/y labels and plot title
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower Bins")

# Bins are 3, so we need 4 numbers
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

# Set group names
group_names = ["Low", "Medium", "High"]

# Apply the function "cut" to determine what each value of "df['horsepower']" belongs to
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)

# Check the first 20 values of the new column "horsepower-binned"
print(df["horsepower-binned"].head(20))

# Note: The code was modified to avoid FutureWarning by not using inplace=True with chained assignment.
# Instead, the replace method was used without inplace=True and the result was assigned back to the original column.

# Indicator variable (or dummy variable)

# Get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])

# Change column names for clarity
dummy_variable_1 = dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'})

# Show first 5 instances of data frame "dummy_variable_1"
print(dummy_variable_1.head(5))

# Merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# Drop original column "fuel-type" from "df"
df = df.drop("fuel-type", axis=1)

aspiration_indicator = pd.get_dummies(df["aspiration"])

aspiration_indicator = aspiration_indicator.rename(columns={"std":"aspiration-std", "turbo":"aspiration-turbo"})

df = pd.concat([df, aspiration_indicator], axis=1)

df = df.drop("aspiration", axis=1)

print(df.head(5))

# Save the new CSV
df.to_csv("automobile.csv", index=False)