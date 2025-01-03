import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"

df = pd.read_csv(path, header=0)

# Linear Regression

lm = LinearRegression()
X = df[['CPU_frequency']]
Y = df['Price']

lm.fit(X, Y)
Yhat = lm.predict(X)

sns.kdeplot(Y, color="r", label="Actual Value")
sns.kdeplot(Yhat, color="b", label="Fitted Values")

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.savefig(fname="bruhlings")
plt.clf()

mse_slr = mean_squared_error(df['Price'], Yhat)
r2_slr = lm.score(X, Y)

# Multiple Linear Regression

lm1 = LinearRegression()
pr = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
lm1.fit(pr, Y)
Y_hat = lm1.predict(pr)

sns.kdeplot(Y, color="r", label="Actual Value")
sns.kdeplot(Y_hat, color="b", label="Fitted Values")

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.savefig(fname="bruhlings")
plt.clf()

mse_mlr = mean_squared_error(Y, Yhat)
r2_mlr = lm1.score(pr, Y)

if mse_slr < mse_mlr and r2_slr > r2_mlr:
    print("SLR is better")
elif mse_mlr < mse_slr and r2_mlr > r2_slr:
    print("MLR is better")
else:
    print("Equal")

# Polynomial Regression

X = X.to_numpy().flatten()

f1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X, Y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X, Y, 5)
p5 = np.poly1d(f5)

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')

PlotPolly(p1, X, Y, "CPU_frequency")
PlotPolly(p3, X, Y, "CPU_frequency")
PlotPolly(p5, X, Y, "CPU_frequency")

r_squared_1 = r2_score(Y, p1(X))
print('The R-square value for 1st degree polynomial is: ', r_squared_1)
print('The MSE value for 1st degree polynomial is: ', mean_squared_error(Y,p1(X)))
r_squared_3 = r2_score(Y, p3(X))
print('The R-square value for 3rd degree polynomial is: ', r_squared_3)
print('The MSE value for 3rd degree polynomial is: ', mean_squared_error(Y,p3(X)))
r_squared_5 = r2_score(Y, p5(X))
print('The R-square value for 5th degree polynomial is: ', r_squared_5)
print('The MSE value for 5th degree polynomial is: ', mean_squared_error(Y,p5(X)))

# Pipeline

stuff = [['scale', StandardScaler()], ['polynomial', PolynomialFeatures(include_bias=False)], ['model', LinearRegression()]]
pipe = Pipeline(stuff)
pr = pr.astype(float)
pipe.fit(pr, Y)
ypipe = pipe.predict(pr)

mse_pipe = mean_squared_error(Y, ypipe)
r2_pipe = r2_score(Y, ypipe)