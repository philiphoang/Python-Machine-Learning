import numpy as np
from sklearn.model_selection import train_test_split
from Tree import Tree
import operator

def learn(X, y, impurity_measure = 'entropy', pruning = False):
    if len(y) == 0:
        return 0

    (X,y) = update_data(X, y)
    print('Impurity_measure:', impurity_measure)

    #X = X_train, y = y_train
    #if pruning:
    #    tree = makeTree(X_train, y_train, impurity_measure)

    return makeTree(X, y, impurity_measure)

def update_data(X, y):
    newX = []
    newy = []
    for row in range(len(y)):
        if '?' not in X[row]:
            newX.append(X[row])
            newy.append(y[row])
    return (newX, newy)

def predict(x, tree):
    check = ''
    for child in tree.children:
        if not child.isLeaf():
            check = x[child.classifier]
            if check == child.name:
                newx = np.delete(x, child.classifier)
                print(newx)
                predict(newx, tree)

    for child in tree.children:
            if check == child.name:
                return child.classifier

    return tree.classifier

def makeTree(X, y, impurity_measure):
    if is_pure(y):
        return Tree(classifier = y[0])

    elif len(np.transpose(X)) == 0: # no features left
        mcl = most_common_label(y)
        return Tree(classifier = mcl)

    else:
        index = calculateInformationGain(X, y, impurity_measure)
        tree = Tree(name = 'branch', classifier = index)

        for attribute_value, [splitted_X, splitted_y] in split(X, y, index).items():
            child = makeTree(splitted_X, splitted_y, impurity_measure)
            child.set_name(attribute_value)
            child.set_data(countLetters(splitted_y))
            tree.add_child(child)

    return tree


def calculateInformationGain(X, y, impurity_measure):
    ig_list = []
    impurity_func = {'entropy': calc_entropy, 'gini': calc_gini}
    measure = impurity_func.get(impurity_measure)

    for row in np.transpose(X):
        probabilities = [counter/len(y) for counter in countLetters(y).values()]
        classification_entropy = measure(probabilities)
        gain = classification_entropy
        for attribute_value, occurrence_dict in zip_xy_class(row, y).items():
            s = sum(occurrence_dict.values())
            X_probabilities = [counter/s for counter in occurrence_dict.values()]
            attribute_entropy = measure(X_probabilities)
            weight = s/len(y)

            gain -= weight * attribute_entropy

        ig_list.append(gain)

    index = np.argmax(ig_list)
    return index

def calc_entropy(listprob):
    entropy = 0
    for prob in listprob:
        if prob != 0:
            entropy += -prob * np.log2(prob)
    return entropy

def calc_gini(listprob):
    gini = 0
    for prob in listprob:
        if prob != 0:
            gini += prob * (1 - prob)
    return gini

def countLetters(array):
    dict = {}
    for i in array:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    return dict

def zip_xy_class(X, y):
    dict = {}
    for attribute, classifier in list(zip(X, y)):
        if attribute in dict:
            if classifier in dict[attribute]:
                dict[attribute][classifier] += 1
            else:
                dict[attribute][classifier] = 1
        else:
            dict[attribute] = {classifier : 1 }
    return dict


def split(X, y, index):
    dict = {}
    for i in range(len(y)):
        if X[i][index] in dict:
            dict[X[i][index]][0].append(X[i][:index] + X[i][index+1:])
            dict[X[i][index]][1].append(y[i])
        else:
            dict[X[i][index]] = [[X[i][:index] + X[i][index+1:]], [y[i]]]
    return dict

def is_pure(y):
    return len(set(y)) == 1

def most_common_label(y):
    dict = {}
    for classifier in y:
        if classifier not in dict.keys():
            dict[classifier] = 0
        dict[classifier] += 1
    sortedClassifier = sorted(dict.items(), key = operator.itemgetter(1), reverse=False)
    return sortedClassifier[0][0]
