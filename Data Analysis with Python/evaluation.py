import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

file = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

df = pd.read_csv(file, header=0)

# only numeric data
df = df._get_numeric_data()

df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis = 1, inplace=True)

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

# split data to training and testing data

y_data = df['price']
x_data = df.drop('price', axis = 1)

# using train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.10, random_state = 1)



print("number of test samples :", x_test.shape[0])
print("number of training samples :", x_train.shape[0])

lre = LinearRegression()

lre.fit(x_train[['horsepower']], y_train)

# get the r^2

print(lre.score(x_test[['horsepower']], y_test))

print(lre.score(x_train[['horsepower']], y_train))

# cross_validation score

# model, features, target, folds
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv = 4)

print(f"The mean of the folds are {Rcross.mean()} and the standard deviation is {Rcross.std()}")

# cross validation to predict output

yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
print(yhat[0:5])

# overfitting and underfitting

lr = LinearRegression()
data = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
lr.fit(x_train[data], y_train)

# prediction
yhat_train = lr.predict(x_train[data])
print(yhat_train[0:5])

yhat_test = lr.predict(x_test[data])
print(yhat_test[0:5])

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

# overfitting

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
print(yhat[0:5])

print(f"Predicted values: {yhat[0:4]}")
print(f"True values: {y_test[0:4].values}")

PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

poly.score(x_train_pr, y_train)

poly.score(x_test_pr, y_test)

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

# poly transformation with > 1 feature:
pr1 = PolynomialFeatures(degree=2)

features = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
x_train_pr1 = pr1.fit_transform(x_train[features])
x_test_pr1 = pr1.fit_transform(x_test[features])

poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)

yhat_test1 = poly1.predict(x_test_pr1)

DistributionPlot(y_test, yhat_test1, "real", "prediction", "Distribuition")

# ridge regression

pr = PolynomialFeatures(degree=2)
features = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']
x_train_pr = pr.fit_transform(x_train[features])
x_test_pr = pr.fit_transform(x_test[features])

RidgeModel = Ridge(alpha=1)

RidgeModel.fit(x_train_pr, y_train)

yhat = RidgeModel.predict(x_test_pr)

print(f"predicted: {yhat[0:4]}")
print(f"test set: {y_test[0:4].values}")

from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

ridgy = Ridge(alpha=10)
ridgy.fit(x_train_pr, y_train)

ridgy.score(x_test_pr, y_test)

# Grid Search

parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
RR = Ridge()
features = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
Grid = GridSearchCV(RR, parameters, cv=4)
Grid.fit(x_data[features], y_data)

BestRR = Grid.best_estimator_

print(BestRR.score(x_test[features], y_test))
