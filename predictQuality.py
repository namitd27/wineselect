#%%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

#%%
from sklearn import linear_model

#%%
wine = pd.read_csv('winequality.csv')

#%%
wine.columns

#%%
response = wine['quality']
featureSet = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
train, test = train_test_split(wine, test_size=0.25, random_state=15, stratify=wine['quality'])
trainX = train[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
trainY = train['quality']
testX = test[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
testY = test['quality']

#%%
train.shape

#%%
#Random Forest Classifier to get Feature Importance
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(trainX, trainY)
pd.Series(rfc.feature_importances_,index=trainX.columns).sort_values(ascending=False)

#%%[markdown]
# From the RFC, it appears that the top 6 features are <br>
# alcohol, volatile acidity, sulphates, total sulphur dioxide, density and chlorides

#%%
# KNN Classifier
neighbors = list(range(1,20))
accuracyScores = []
for n in neighbors:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(trainX, trainY)
    prediction = model.predict(testX)
    #print("With "+str(n)+" neigbours, accuracy = "+str(metrics.accuracy_score(prediction, testY)))
    accuracyScores.append(metrics.accuracy_score(prediction, testY))
plt.plot(neighbors, accuracyScores)
plt.xticks(neighbors)
plt.grid(True)
plt.show()

#%%
# Using all the feautures provided
accuracyScores = []
classifiers = ['Linear Svm','Radial Svm','Logistic Regression','Decision Tree']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(trainX,trainY)
    prediction = model.predict(testX)
    accuracyScores.append(metrics.accuracy_score(prediction,testY))
models_dataframe = pd.DataFrame(accuracyScores,index=classifiers)   
models_dataframe.columns = ['Accuracy']
models_dataframe

#%%
# Lasso
model = linear_model.Lasso(alpha = 0.001)
model.fit(trainX, trainY)
prediction = model.predict(testX)
print("Lasso Accuracy: ", model.score(testX,testY))

#%%
#Ridge
model = linear_model.Ridge(alpha = 0.05, normalize=True)
model.fit(trainX, trainY)
prediction = model.predict(testX)
print("Ridge Accuracy: ", model.score(testX,testY))
