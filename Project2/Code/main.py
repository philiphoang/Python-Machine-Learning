import numpy as np 
import random as rn
import tensorflow as tf
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Tools for measure performance 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

# Functions from local files
from readhandwrittendigits import readhandwrittendigits
from neuralnet import neural_network_training

def fit_and_predict(clf, X_train, X_test, y_train, y_test):
    print("Starting " , clf.__class__.__name__)
    start_time = time.time()   
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Time", clf.__class__.__name__, ": %0.10f seconds" % (time.time() - start_time))
    print("Accuracy: %0.4f " % accuracy_score(y_test, y_pred))
    
    conf_matrix(clf.__class__.__name__, y_test, y_pred)

def randomforest(estimators = 45, features = 400):
    rnd_clf = RandomForestClassifier(random_state=42, n_estimators=estimators, max_features=features, n_jobs = -1)
    return rnd_clf

def naive_bayes(X_train, X_test, y_train, y_test):
    # Naive Bayes Classifications
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()

    # Must convert datatypes to float to run Naive Bayes Class
    X_train = np.array(X_train).astype(np.float)
    y_train = np.array(y_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    y_test = np.array(y_test).astype(np.float)

    fit_and_predict(gnb, X_train, X_test, y_train, y_test)
    fit_and_predict(mnb, X_train, X_test, y_train, y_test)
    fit_and_predict(bnb, X_train, X_test, y_train, y_test)

def gridSearchRandomForest(clf, X_train, X_test, y_train, y_test):
    grid = [{
        "n_estimators": [150, 300, 400, 500, 700],
        "max_features": [25, 35, 45, 55, 70]
    }]

    grid_clf = GridSearchCV(clf, grid, cv=5, n_jobs = 1)
    fit_and_predict(grid_clf, X_train, X_test, y_train, y_test)
    print(grid_clf.best_score_)
    print(grid_clf.best_params_)

def gridSearchNeuralNetwork(clf, X_train, X_test, y_train, y_test):
    grid = [{
        "epochs": [5, 10, 15, 20, 25],
        "batch_size": [500, 750, 1000, 1250, 1500]
    }]

    grid_clf = GridSearchCV(clf, grid, cv=5, n_jobs = 1)
    fit_and_predict(grid_clf, X_train, X_test, y_train, y_test)
    print(grid_clf.best_score_)
    print(grid_clf.best_params_)

def conf_matrix(name, y_test, y_pred):
    target_names = [str(x) for x in range(0, 10)]
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels= target_names,
            yticklabels= target_names)

    print(classification_report(y_test, y_pred, target_names=target_names))
    plt.title(name)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

tf.set_random_seed(19)
rn.seed(19)
np.random.seed(19)

start_time = time.time()
X_train, X_test, y_train, y_test = readhandwrittendigits()
print("Time reading data: %0.10f seconds" % (time.time() - start_time))

y_train = np.ravel(y_train)

# --- Classification model: Neural Network ---
#nn_clf = neural_network_training(X_train)
#gridSearchNeuralNetwork(nn_clf,  X_train, X_test, y_train, y_test)
#fit_and_predict(nn_clf, X_train, X_test, y_train, y_test)

# --- Classification model: Random Forest  ---
rnd_clf = randomforest()
#gridSearchRandomForest(rnd_clf, X_train, X_test, y_train, y_test)
fit_and_predict(rnd_clf, X_train, X_test, y_train, y_test)

# --- Classification models: ---
#naive_bayes(X_train, X_test, y_train, y_test)