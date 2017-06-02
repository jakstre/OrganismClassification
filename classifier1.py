import numpy as np

#import matplotlib.pyplot as plt
import nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import features

loadDataFromFile = False
runOnlyNN = False


def fprint(*output):

    ostring = str(output[0])
    for i in range(1,len(output)):
        ostring+= " " + str(output[i])

    print (ostring)
    with open("results.txt", "a") as f:
        f.write("{}\n".format(ostring))


def run():
    data = None
    labels = None
    labelsOneHot = None
    if (loadDataFromFile):
        data, labels, labelsOneHot = features.loadData()
        print "features were loaded"
    else:
        data, labels, labelsOneHot = features.prepareData(features.get_features)
        print "features were prepared"
    print(data.shape, labels.shape, labelsOneHot.shape)
    #xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.33, random_state=42, stratify = labels)
    #_,_,yTrainOneHot, yTestOneHot = train_test_split(data, labelsOneHot, test_size=0.33, random_state=42, stratify = labels)

    skf = StratifiedKFold(n_splits=5)

    # cpu is probably good enough
    #with tf.device("/cpu:0"):

    # try more hidden layers
    
    scores = []
    classifier = nn.MLP([data.shape[1], 1000, labelsOneHot.shape[1]])
    for trainIndex, testIndex in skf.split(data , labels):
        xTrain, xTest = data[trainIndex], data[testIndex]
        yTrainOneHot, yTestOneHot = labelsOneHot[trainIndex], labelsOneHot[testIndex]
        score = classifier.fit(xTrain, yTrainOneHot, xTest, yTestOneHot, save = False, verbose = False) 
        print (score)
        scores.append(score)
    fprint ("Neural network score mean: " , np.mean(scores))
    fprint ("Neural network score std: " ,  np.std(scores))
    fprint ("Neural network score max/min:" , np.max(scores), np.min(scores))
    fprint ("")


    if runOnlyNN:
        return
    
    classifier = LogisticRegression()
    scores = []
    for trainIndex, testIndex in skf.split(data , labels):
        xTrain, xTest = data[trainIndex], data[testIndex]
        yTrain, yTest = labels[trainIndex], labels[testIndex]
        classifier.fit(xTrain, yTrain)
        score = classifier.score(xTest, yTest)
        print (score)
        scores.append(score)
    fprint ("Logistic regression score mean: " , np.mean(scores))
    fprint ("Logistic regression score std: " ,  np.std(scores))
    fprint ("Logistic regression score max/min:", np.max(scores), np.min(scores))
    fprint ("")

    # SVM yields poor results
    """
    classifier = svm.SVC()
    scores = []
    for trainIndex, testIndex in skf.split(data , labels):
        xTrain, xTest = data[trainIndex], data[testIndex]
        yTrain, yTest = labels[trainIndex], labels[testIndex]
        classifier.fit(xTrain, yTrain)
        score = classifier.score(xTest, yTest)
        print (score)
        scores.append(score)
    fprint ("SVM score mean: " , np.mean(scores))
    fprint ("SVM score std: " ,  np.std(scores))
    fprint ("SVM score max/min:", np.max(scores), np.min(scores))
    fprint ("")
    """

    classifier = RandomForestClassifier(n_estimators=100)
    scores = []
    for trainIndex, testIndex in skf.split(data , labels):
        xTrain, xTest = data[trainIndex], data[testIndex]
        yTrain, yTest = labels[trainIndex], labels[testIndex]
        classifier.fit(xTrain, yTrain)
        score = classifier.score(xTest, yTest)
        print (score)
        scores.append(score)
    fprint ("Random forest score mean: " , np.mean(scores))
    fprint ("Random forest score std: " ,  np.std(scores))
    fprint ("Random forest score max/min:", np.max(scores), np.min(scores))
    fprint ("")

if __name__ == "__main__":
    run()
    


