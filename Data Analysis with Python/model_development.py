import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Linear Regression modules
from sklearn.linear_model import LinearRegression

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Pipeline
from sklearn.pipeline import Pipeline

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df = pd.read_csv(file_path)

# Create a linear regression object
lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']

# Fit the linear model using highway-mpg
lm.fit(X,Y)

# Output a prediction
Yhat=lm.predict(X)
#print(Yhat[0:5])

# Output the intercept
#print(lm.intercept_)

# Output the slope
#print(lm.coef_)

# Model Evaluation using Visualization

# Regression Plot

width = 12
height = 10
plt.figure(figsize=(width, height))

sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
plt.savefig(fname="highway-mpg_price.png")

# peak RPM

plt.clf()
plt.figure(figsize=(width, height))
sns.regplot(x='peak-rpm', y='price', data=df)
plt.ylim(0,)
plt.savefig(fname="peak-rpm_price.png")

print(df[["peak-rpm","highway-mpg","price"]].corr())

# Residual Plot
plt.clf()
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.savefig(fname="residual_plot.png")

# Multiple Linear Regression

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
Y_hat = lm.predict(Z)

plt.clf()
plt.figure(figsize=(width, height))

ax1 = sns.histplot(df['price'], kde=True, color='r', label='Actual Value', stat="density", element="step")
sns.histplot(Y_hat, kde=True, color='b', label='Fitted Values', stat="density", element="step", ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.savefig(fname="actual_vs_fitted.png")

# Polynomial Regression and Pipelines

def PlotPoly(model, independent_variable, dependent_variable, Name):
    """
    Function to plot the polynomial regression
    """
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.savefig(fname="poly_fit.png")

# Get the variables
x = df['highway-mpg']
y = df['price']

# Fit the polynomial using the function polyfit, then use the function poly1d to display the polynomial function
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
#print(p)

PlotPoly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)

f = np.polyfit(x, y, 11)
p = np.poly1d(f)

PlotPoly(p, x, y, 'highway-mpg')

pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)

# Pipeline

input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]

pipe = Pipeline(input)

pipe.fit(Z, y)

ypipe = pipe.predict(Z)
print(ypipe[0:4])

input = [('scale', StandardScaler()), ('model', LinearRegression())]
pipe = Pipeline(input)
pipe.fit(Z, y)

ypipe = pipe.predict(Z)


# Measures for In-Sample Evaluation

# Mean Squared Error

# Simple Linear Regression
lm.fit(X, Y)
print('The R-square is: ', lm.score(X, Y))

Yhat = lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)

print('The mean square error of price and predicted value is: ', mse)

# Multiple Linear Regression

lm.fit(Z, df['price'])

Y_predict_multifit = lm.predict(Z)

print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_predict_multifit))

# Polynomial Fit

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

mean_squared_error(df['price'], p(x))

# Prediction and Decision Making

new_input = np.arange(1, 100, 1).reshape(-1, 1)

lm.fit(X, Y)
yhat = lm.predict(new_input)
plt.clf()
plt.figure(figsize=(width, height))
plt.plot(new_input, yhat)
plt.savefig(fname="new_input.png")
