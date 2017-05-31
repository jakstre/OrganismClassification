import numpy as np
import os
import fastaParser

basesMap = { 'w':'a',
             's':'c',
             'm':'a',
             'k':'g',
             'r':'a',
             'y':'c',
             'b':'c',
             'd':'a',
             'h':'a',
             'v':'a',
             'n':'a'}

bases = ['a', 't', 'g', 'c']

basesNumerical = { 'a': 0.0,
                   't': 0.25,
                   'g': 0.5,
                   'c': 1.0}

def baseToNumber(base):
    x = base
    if base not in bases:
        x = basesMap[base]
    return basesNumerical[x];


def emptyPairs():
    d = {}

    for i in bases:
        for j in bases:
                d[i+j]=0.0
    return d

def pairHist(sequence):
    pairs = emptyPairs()
    l = len(sequence)
    a = sequence[0]
    b = sequence[1]
    

    if a not in bases:
        if a not in basesMap:
            print("Unkown character:", a)
        else:
            a = basesMap[a]


    for i in range(l-1):
        b = sequence[i+1]
        if b not in bases:
            if b not in basesMap:
                print("Unkown character:", b)
            else:
                b = basesMap[b]

        d = a+b
        if d not in pairs:
            print("Unknown digram:", d)
        else:
            pairs[d]+=1
        a = b

    for key in pairs:
        pairs[key]/=l-1
    return np.asarray(list(pairs.values()))


def emptyTriplets():
    d = {}
    for i in bases:
        for j in bases:
            for k in bases:
               d[i+j+k]=0.0
    return d
        
def tripletsHist(sequence):
    triplets = emptyTriplets()
    l = len(sequence)
    a = sequence[0]
    b = sequence[1]
    c = sequence[2]

    if a not in bases:
        if a not in basesMap:
            print("Unkown character:", a)
        else:
            a = basesMap[a]

    if b not in bases:
        if b not in basesMap:
            print("Unkown character:", b)
        else:
            b = basesMap[b]

    for i in range(l-2):
        c = sequence[i+2]
        if c not in bases:
            if c not in basesMap:
                print("Unkown character:", c)
            else:
                c = basesMap[c]

        d = a+b+c
        if d not in triplets:
            print("Unknown trigram:" ,d)
        else:
            triplets[d]+=1
        a = b
        b = c

    for key in triplets:
        triplets[key]/=l-2
#     return triplets
    return np.asarray(list(triplets.values()))


def fourier(sequence, n=50):
    seq = np.array(list(map(baseToNumber, sequence)))
    freq = np.fft.rfft(seq)
    freqSlice = freq[:n]#freq[-n:]
    # concatenate real and imaginary parts to single vector
    return np.append(freqSlice[:].real,freqSlice[:].imag)
     



def oneHot(names):
    d = {}
    i = 0
    for name in names:
        code = np.zeros(len(names), dtype = np.float32)
        code[i] = 1.0
        d[name] = code
        i+=1
    return d

def integerCode(names):
    d = {}
    i = 0
    for name in names:
        d[name] = i#np.array([i],dtype = np.float32)
        i+=1
    return d

dataDir = "./data"

def prepareData(featureExtractor, save = True, fastaName = 'current_Fungi_unaligned.fa', numSpecies = 10):
    data = []
    labels = []
    labelsOneHot = []
    
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    with open(fastaName) as fp:
        namesSeqMap = fastaParser.getNamesSeqSet(fp)
        print ("Parsed:", len(namesSeqMap), "species")

        sortedBySeqCount = sorted(namesSeqMap.items(), key=lambda x: len(x[1]), reverse = True)

        print(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))

        #codes = None
        #if oneHot:
        #    codes = oneHot(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))
        #else:
        #    codes = integerCode(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))

        intCodes  = integerCode(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))
        oneHotCodes =  oneHot(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))

        i = 0
        for pair in sortedBySeqCount:
            for seq in pair[1]:
                data.append(featureExtractor(seq))
                labels.append(intCodes[pair[0]])
                labelsOneHot.append(oneHotCodes[pair[0]])
            i+=1
            if i==numSpecies:
                break

    d = np.asarray(data)
    l = np.asarray(labels)
    loh = np.asarray(labelsOneHot)
    if save:
        np.save(dataDir + "/x.npy",d)
        np.save(dataDir + "/y.npy",l)
        np.save(dataDir + "/yoh.npy",loh)
    return d,l, loh

def loadData():
    data = np.load(dataDir + "/x.npy")
    labels = np.load(dataDir + "/y.npy")
    labelsOneHot = np.load(dataDir + "/yoh.npy")
    return data, labels, labelsOneHot
