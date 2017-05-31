import numpy as np

#import matplotlib.pyplot as plt
import nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import features

loadDataFromFile = False



if __name__ == "__main__":
    data = None
    labels = None
    labelsOneHot = None
    if (loadDataFromFile):
        data, labels, labelsOneHot = features.loadData()
    else:
        data, labels, labelsOneHot = features.prepareData(features.fourier)
    print(data.shape, labels.shape, labelsOneHot.shape)
    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.33, random_state=42, stratify = labels)
    _,_,yTrainOneHot, yTestOneHot = train_test_split(data, labelsOneHot, test_size=0.33, random_state=42, stratify = labels)

    # cpu is probably good enough
    #with tf.device("/cpu:0"):

    # try more hidden layers
    classifier = nn.MLP([xTrain.shape[1], 50, yTrainOneHot.shape[1]])
    nn.NNfit(classifier, xTrain, yTrainOneHot, xTest, yTestOneHot) 
    nnScore = nn.NNtest(classifier, xTest,yTestOneHot)
    print ("Neural network score:", nnScore)


    classifier2 = LogisticRegression()
    classifier2.fit(xTrain, yTrain)
    print("Logistic regression score: ",classifier2.score(xTest, yTest))


