import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

path = "/content/drive/MyDrive/file.csv"
df = pd.read_csv(path)

print(df.isnull().sum())

print(df.describe)

X=df.drop(['Occupancy'],axis=1)
y=df['Occupancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('test mae',metrics.mean_absolute_error(y_test,y_pred))
print('test r^2',metrics.r2_score(y_pred,y_test))
print('test mape',metrics.mean_absolute_percentage_error(y_test,y_pred))
print('test rmse',sqrt(metrics.mean_squared_error(y_test,y_pred)))

y_train_pred = regressor.predict(X_train)
print('train mae',metrics.mean_absolute_error(y_train,y_train_pred))
print('tain r^2',metrics.r2_score(y_train_pred,y_train))
print('train mape',metrics.mean_absolute_percentage_error(y_train,y_train_pred))
print('train rmse',sqrt(metrics.mean_squared_error(y_train,y_train_pred)))

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

"""its possibile that we could get an over fitting after makjng degree 3"""

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, y, test_size = 0.2)

"""dataset shows high multicollinearity or you want to automate variable elimination and feature selection use lasso,ridge when a dataset contains a higher number of predictor variables than observations.lasso produces simpler and more interpretable models that include only a reduced set of the predictors"""

lr = LinearRegression()
lr.fit(Xp_train, yp_train)
rr = Ridge(alpha=0) 
rr.fit(Xp_train, yp_train)

lasso = Lasso()
ll = Lasso(tol=0.1)
ll.fit(Xp_train,yp_train)

pred_train_rr= rr.predict(Xp_train)
print('ridge train r^2',metrics.r2_score(yp_train, pred_train_rr))
print('ridge train mae',metrics.mean_absolute_error(yp_train,pred_train_rr))
print('ridge train mape',metrics.mean_absolute_percentage_error(yp_train,pred_train_rr))
print('ridge train rmse',sqrt(metrics.mean_squared_error(yp_train,pred_train_rr)))
pred_test_rr= rr.predict(Xp_test)
print('ridge test r^2',metrics.r2_score(yp_test, pred_test_rr))
print('ridge test mae',metrics.mean_absolute_error(yp_test,pred_test_rr))
print('ridge test mape',metrics.mean_absolute_percentage_error(yp_test,pred_test_rr))
print('ridge test  rmse',sqrt(metrics.mean_squared_error(yp_test,pred_test_rr)))

pred_train_lasso= ll.predict(Xp_train)
print('lasso train r^2',metrics.r2_score(yp_train, pred_train_lasso))
print('lasso train mae',metrics.mean_absolute_error(yp_train,pred_train_lasso))
print('lasso train mape',metrics.mean_absolute_percentage_error(yp_train,pred_train_lasso))
print('lasso train rmse',sqrt(metrics.mean_squared_error(yp_train,pred_train_lasso)))
pred_test_lasso= ll.predict(Xp_test)
print('lasso test r^2',metrics.r2_score(yp_test, pred_test_lasso))
print('lasso test mae',metrics.mean_absolute_error(yp_test,pred_test_lasso))
print('lasso test mape',metrics.mean_absolute_percentage_error(yp_test,pred_test_lasso))
print('lasso test rmse',sqrt(metrics.mean_squared_error(yp_test,pred_test_lasso)))

# create regressor object
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)
# fit the regressor with x and y data
rfr.fit(X_train, y_train)

y_test_pred = rfr.predict(X_test)
y_train_pred = rfr.predict(X_train)
print('forest test rmse',sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('forest test r^2',metrics.r2_score(y_test, y_test_pred))
print('forest test mae',metrics.mean_absolute_error(y_test, y_test_pred))
print('forest test mape',metrics.mean_absolute_percentage_error(y_test, y_test_pred))

print('forest train rmse',sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('forest train r^2',metrics.r2_score(y_train, y_train_pred))
print('forest train mae',metrics.mean_absolute_error(y_train, y_train_pred))
print('forest train mape',metrics.mean_absolute_percentage_error(y_train, y_train_pred))

"""the best model is random forest reggression having the best rmse,r^2 and mae showing that the model is not in under or over fitting but also works perfectly.
there is no over under fitting in any model 
"""
