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
from sklearn import linear_model

#%%
from sklearn.naive_bayes import GaussianNB

#%%
wine = pd.read_csv('refinedWineQuality.csv')

#%%
wine.drop('Unnamed: 0', axis = 1, inplace=True)

#%%
wine.columns[:11]

#%%
#Standardization
response = wine['Rating']
featureSet = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
features_standard = StandardScaler().fit_transform(featureSet)
stdWine = pd.DataFrame(features_standard, columns=[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']])
stdWine['Rating'] = response
train, test = train_test_split(stdWine, test_size=0.25, random_state=15, stratify=stdWine['Rating'])
trainX = train[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
trainY = train['Rating']
testX = test[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
testY = test['Rating']

#%%
testX.columns

#%%
#Random Forest Classifier to get Feature Importance
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(trainX, trainY)
pd.Series(rfc.feature_importances_,index=trainX.columns).sort_values(ascending=False)


#%%
newTrainX = train[['alcohol', 'volatile acidity', 'sulphates', 'total sulfur dioxide', 'density', 'chlorides']]
newTestX = test[['alcohol', 'volatile acidity', 'sulphates', 'total sulfur dioxide', 'density', 'chlorides']]

#%%
# KNN Classifier
neighbors = list(range(1,11))
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
classifiers = ['Linear SVC','Radial SVC','Logistic Regression','Decision Tree','KNN']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=9)]
for i in models:
    model = i
    model.fit(trainX,trainY)
    prediction = model.predict(testX)
    accuracyScores.append(metrics.accuracy_score(prediction,testY))
    #accuracyScores.append(model.score(testX, testY))
models_dataframe = pd.DataFrame(accuracyScores,index=classifiers)   
models_dataframe.columns = ['Accuracy']
models_dataframe


#%%
# Using all the feautures provided
accuracyScores = []
classifiers = ['Linear SVC','Radial SVC','Logistic Regression','Decision Tree','KNN']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=9)]
cnt = 0 
for i in models:
    model = i
    model.fit(trainX,trainY)
    prediction = model.predict(testX)
    print("Model: ",classifiers[cnt],metrics.classification_report(testY,prediction))
    cnt = cnt+1
    #accuracyScores.append(model.score(testX, testY))


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

#%%
kfold = KFold(n=10,random_state=10)

#%%
cvMean = []
results = []
classifiers = ['Linear Svm','Radial Svm','Logistic Regression','Decision Tree','KNN']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=3)]
for i in models:
    model = i
    result = cross_val_score(model, wine[wine.columns[:11]], wine['quality'],cv=kfold, scoring='accuracy')
    results.append(result)
    cvMean.append(result.mean())
new_models_df = pd.DataFrame(cvMean, index=classifiers)
new_models_df.columns = ['CV Mean']
new_models_df

#%%
# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(trainX,trainY)
prediction = gnb.predict(testX)
print("Accuracy: ",metrics.accuracy_score(prediction,testY))

#%%
# Recursive Feature Elimination with Cross Validation
classifiers = ['Decision Tree']
models = [DecisionTreeClassifier()]
for i in models:
    model = i
    rfecv = RFECV(model, step=1, cv=StratifiedKFold(2),scoring='accuracy')
    rfecv.fit(wine[wine.columns[:11]], wine['Rating'])
    print("Optimal number of features :" + str(rfecv.ranking_))


#%%
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#%%
# Ensemble: Combine Radial SVC and Decision Tree
radial_svc = svm.SVC(kernel='rbf', C=0.5, gamma=10, probability=True)
dtc = DecisionTreeClassifier()
ensemble_radial_dtc = VotingClassifier(estimators=[('Radial SVC', radial_svc),('Decision Tree Classifier',dtc)], voting='soft', weights=[10,2])
ensemble_radial_dtc.fit(trainX,trainY)
prediction = ensemble_radial_dtc.predict(testX)
print("Accuracy of Radial SVC and Decision Tree Classifier: ", metrics.accuracy_score(prediction, testY))

#%%
radial_svc = svm.SVC(kernel='rbf', C=0.5, gamma=10, probability=True)
dtc = DecisionTreeClassifier()
ensemble_radial_dtc = VotingClassifier(estimators=[('Radial SVC', radial_svc),('Decision Tree Classifier',dtc)], voting='soft', weights=[2,1])
ensemble_radial_dtc.fit(newTrainX,trainY)
prediction = ensemble_radial_dtc.predict(newTestX)
print("Accuracy of Radial SVC and Decision Tree Classifier: ", metrics.accuracy_score(prediction, testY))

#%%
# Ensemble: Combine Linear SVC and Decision Tree
linear_svc = svm.SVC(kernel='linear', C=0.5, gamma=10, probability=True)
dtc = DecisionTreeClassifier()
ensemble_radial_dtc = VotingClassifier(estimators=[('Linear SVC', linear_svc),('Decision Tree Classifier',dtc)], voting='soft', weights=[1,5])
ensemble_radial_dtc.fit(trainX,trainY)
prediction = ensemble_radial_dtc.predict(testX)
print("Accuracy of Linear SVC and Decision Tree Classifier: ", metrics.accuracy_score(prediction, testY))


#%%
# Ensemble: Combine Radial, DTC and Logistic
logReg = LogisticRegression(C=0.5)
ensemble_radial_dtc_log = VotingClassifier(estimators=[('Radial SVC', radial_svc),('Decision Tree Classifier',dtc),('Logistic',logReg)],voting='soft',weights=[2,3,1])
ensemble_radial_dtc_log.fit(trainX, trainY)
prediction = ensemble_radial_dtc_log.predict(testX)
print("Accuracy: ", metrics.accuracy_score(prediction, testY))