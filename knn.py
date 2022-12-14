import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.layers import Activation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

path = "/content/drive/MyDrive/winequality-red.csv"
df = pd.read_csv(path)
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df = df.drop('quality', axis=1)
print(df.isnull().sum())
print(df.describe())
X = df.drop(['best quality'], axis=1)
y = df['best quality']

y.value_counts().plot(kind='bar')
print(y.value_counts().sum())

from imblearn.over_sampling import SMOTE
os = SMOTE()
X, y = os.fit_resample(X, y)

"""there was no need for resampling the data because there was no underfitting or iverfitting but we choose to use smote overfitiing because we dont want to lose data and adding little data wont make a big impact on the dataset"""

y.value_counts().plot(kind='bar')
print(y.value_counts().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=123,test_size=0.2)

model = xgb.XGBClassifier()

pipeline = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('model', model)
])

param_grid = {
    'pca__n_components': [5, 10, 15, 20, 25, 30],
    'model__n_estimators': [100, 300, 200,400,500,800]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid2 = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

grid.fit(X_train, y_train)
grid2.fit(X_test, y_test)

mean_score = grid.cv_results_["mean_test_score"][grid.best_index_]
std_score = grid.cv_results_["std_test_score"][grid.best_index_]

grid.best_params_, mean_score, std_score

print(f"Best parameters: {grid.best_params_}")
print(f"Mean CV score: {mean_score: .6f}")
print(f"Standard deviation of CV score: {std_score: .6f}")

"""1)model n_estimators is used as  the number of runs XGBoost will try to learn
2)PCA is used to reduce the dimensionality of the financial input data set and the DWT is used to perform a noise reduction to every feature.
*******************************
 The resultant data set is then fed to an XGBoost binary classifier that has its hyperparameters optimized 

"""

mean_score1 = grid2.cv_results_["mean_test_score"][grid2.best_index_]
std_score1 = grid2.cv_results_["std_test_score"][grid2.best_index_]

grid2.best_params_, mean_score1, std_score1

print(f"Best parameters: {grid2.best_params_}")
print(f"Mean CV score: {mean_score1: .6f}")
print(f"Standard deviation of CV score: {std_score1: .6f}")

#Normalizing the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
y_train=y_train.values.reshape(-1,1)
y_train= sc.fit_transform(y_train)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()

print(X_train.shape)

print("X train shape:",X_train.shape)
print("y train shape:",y_train.shape)
print("X test shape:",X_test.shape)
print("y test shape:",y_test.shape)

X,X1,y,y1 = train_test_split(X_train,y_train,test_size = 0.1)
print("X train shape:",X1.shape)
print("y train shape:",y1.shape)
print("X test shape:",X.shape)
print("y test shape:",y.shape)

# define the keras model
model = Sequential()
model.add(Dense(9,input_dim=11, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#fit the keras model on the dataset
model.fit(X, y, epochs=350, batch_size=10,verbose=2)
# evaluate the keras model
accuracy = model.evaluate(X, y)
# Print a summary of the Keras model:
model.summary()

model.fit(X1, y1, epochs=350, batch_size=64,verbose=4)
# evaluate the keras model
accuracy = model.evaluate(X1, y1)

"""The first is input layer has 9 nodes and uses the relu activation function. 
The second hidden layer has 7 nodes and uses the relu activation function.
The output layer has 2 node and uses the sigmoid activation function.
the neteork have 194 params
"""

y_pred = model.predict(X)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y)):
    test.append(np.argmax(y[i]))
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

y_pred = model.predict(X1)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y1)):
    test.append(np.argmax(y1[i]))
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
