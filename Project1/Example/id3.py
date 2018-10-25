import operator
from math import log

def entropy(data):
    entries = len(data)
    attributes = {}
    for feature in data:
        attribute = feature[-1]
        if attribute not in attributes.keys():
            attributes[attribute] = 0
            attributes[attribute] += 1
        entropy = 0.0
        for key in attributes:
            probability = float(attributes[key]) / entries
            entropy -= probability * log(probability,2)
        return entropy

def split(data, axis, val):
    splittedData = []
    for feature in data:
        if feature[axis] == val:
            reducedFeature = feature[:axis]
            reducedFeature.extend(feature[axis+1])
            splittedData.append(reducedFeature)
    print(splittedData)
    return splittedData

def bestInformationGain(data):
    features = len(data[0]) - 1
    baseEntropy = entropy(data)
    bestIG = 0.0
    bestFeature = -1

    for i in range(features):
        featureList = [ex[i] for ex in data]
        uniqueValues = set(featureList)
        newEntropy = 0.0
        for value in uniqueValues:
            splittedData = split(data, i, value)
            probability = len(splittedData)/float(len(data))
            newEntropy += probability * entropy(splittedData)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestIG):
            bestIG = infoGain
            bestFeature = i
    return bestFeature

def majority(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def tree(data, labels):
    classList = [ex[-1] for ex in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:
        return majority(classList)
    bestFeat = bestInformationGain(data)
    bestFeatLabel = labels[bestFeat]
    theTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [ex[bestFeat] for ex in data]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        theTree[bestFeatLabel][value] = tree(split(data, bestFeat, value),subLabels)
    return theTree
