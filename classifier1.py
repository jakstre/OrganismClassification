import numpy as np
import matplotlib.pyplot as plt
#import nn
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import fastaParser
import features



def oneHot(names):
    d = {}
    i = 0
    for name in names:
        code = np.zeros(len(names), dtype = np.float32)
        code[i] = 1.0
        d[name] = code
        i+=1
    return d

def prepareData(fastaName = 'current_Fungi_unaligned.fa', numSpecies = 10, encode = True):
    data = []
    labels = []
    with open(fastaName) as fp:
        namesSeqMap = fastaParser.getNamesSeqSet(fp)
        print ("Parsed:", len(namesSeqMap), "species")

        sortedBySeqCount = sorted(namesSeqMap.items(), key=lambda x: len(x[1]), reverse = True)

        print(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))

        if encode:
            oneHotCodes = oneHot( list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))


        i = 0
        for pair in sortedBySeqCount:
            for seq in pair[1]:
                data.append(features.tripletsHist(seq))
                if encode:
                    labels.append(oneHotCodes[pair[0]])
                else:
                    labels.append(pair[0])
            i+=1
            if i==numSpecies:
                break

    return np.asarray(data), (np.asarray(labels) if encode else labels) 



if __name__ == "__main__":
    data, labels = prepareData(encode = False)
    #print(data.shape, labels.shape)
    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.33, random_state=42, stratify = labels)
    classifier = LogisticRegression()
    classifier.fit(xTrain, yTrain)
    print("Score: ",classifier.score(xTest, yTest))