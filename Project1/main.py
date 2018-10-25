import csv
import numpy as np
from ID3 import learn, makeTree, predict
from Tree import Tree

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X = []
y = []

with open('agaricus-lepiota.data') as csv_file:
        reader = csv.reader(open("agaricus-lepiota.data", "rb"), delimiter=",")
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            y.append(row[0])
            X.append(row[1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


#For my ID3 implementation:
myTree = learn(X_train, y_train, 'entropy', True)
myTree.show()

dict = {}
for row in range(len(X_test)):
    pred = predict(X[row], myTree)
    result = pred == y_test[row]
    if result not in dict:
        dict[result] = 1

    if (result):
        dict[result] += 1
    else:
        dict[result] += 1
print(dict)


#5. Implementation comparision
X_T = []
le = LabelEncoder()
for i in range(len(np.transpose(X_train))):
    X_T.append(le.fit_transform(np.transpose(X_train)[i]))
X_train = np.transpose(X_T)

X_T = []
for i in range(len(np.transpose(X_train))):
    X_T.append(le.fit_transform(np.transpose(X_test)[i]))
X_test = np.transpose(X_T)

y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)


dtc = tree.DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train, y_train)
dtc_predict = dtc.predict(X_test)

dict = {}
for i in dtc_predict:
    result = dtc_predict[i] == y_test[i]
    if result not in dict:
        dict[result] = 1

    if (result):
        dict[result] += 1
    else:
        dict[result] += 1

print(dict)

#print(prune(X, y, tree))
