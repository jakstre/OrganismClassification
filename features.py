import numpy as np


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