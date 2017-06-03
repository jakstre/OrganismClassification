import numpy as np
import os
import fastaParser
from scipy import signal
import matplotlib.pyplot as plt
from scipy.misc import imresize
import pywt


basesMap = {'w': 'a',
            's': 'c',
            'm': 'a',
            'k': 'g',
            'r': 'a',
            'y': 'c',
            'b': 'c',
            'd': 'a',
            'h': 'a',
            'v': 'a',
            'n': 'a'}

bases = ['a', 't', 'g', 'c']

basesNumerical = {'a': 0.0,
                  't': 0.25,
                  'g': 0.5,
                  'c': 1.0}


def get_features(sequence):
    sequence = to_pure_atgc(sequence)

    feats = []
    # feats = np.hstack((feats, singles_hist(sequence)))
    # feats = np.hstack((feats, pairHist(sequence)))
    # feats = np.hstack((feats, tripletsHist(sequence)))
    feats = np.hstack((feats, fourier(sequence)))
    # feats = np.hstack((feats, singlet_transitions(sequence)))
    # feats = np.hstack((feats, doublet_transitions(sequence)))
    feats = np.hstack((feats, triplet_transitions(sequence)))
    feats = np.hstack((feats, wavelet(sequence)))

    return feats


def to_pure_atgc(sequence):
    pure_sequence = ""
    for i in range(len(sequence)):
        c = sequence[i]
        if c not in bases:
            if c not in basesMap:
                print("Unknown character:", c)
            else:
                pure_sequence += basesMap[c]
        else:
            pure_sequence += c
    return pure_sequence


def triplet_transitions(sequence):
    to_index = {}
    seq_len = len(sequence)
    bases_len = len(bases)

    for i in range(bases_len):
        to_index[bases[i]] = i

    mat = np.zeros((bases_len * bases_len * bases_len, bases_len))
    hits = np.zeros((mat.shape[0],))
    a = sequence[0]
    b = sequence[1]
    c = sequence[2]

    for i in range(seq_len - 3):
        d = sequence[i + 3]
        mat[to_index[a] * bases_len * bases_len + to_index[b] * bases_len + to_index[c]][to_index[d]] += 1
        hits[to_index[a] * bases_len * bases_len + to_index[b] * bases_len + to_index[c]] += 1
        a = b
        b = c
        c = d

    for i in range(mat.shape[0]):
        # The correct way
        # divisor = hits[i]
        # Totally wrong, but works better!
        divisor = seq_len
        if divisor == 0:
            continue
        for j in range(mat.shape[1]):
            mat[i][j] /= divisor

    return np.array(mat).flatten()


def doublet_transitions(sequence):
    to_index = {}
    seq_len = len(sequence)
    bases_len = len(bases)

    for i in range(bases_len):
        to_index[bases[i]] = i

    mat = np.zeros((bases_len * bases_len, bases_len))
    hits = np.zeros((mat.shape[0],))
    a = sequence[0]
    b = sequence[1]

    for i in range(seq_len - 2):
        c = sequence[i + 2]
        mat[to_index[a] * bases_len + to_index[b]][to_index[c]] += 1
        hits[to_index[a] * bases_len + to_index[b]] += 1
        a = b
        b = c

    for i in range(mat.shape[0]):
        # The correct way
        # divisor = hits[i]
        # Totally wrong, but works better!
        divisor = seq_len
        if divisor == 0:
            continue
        for j in range(mat.shape[1]):
            mat[i][j] /= divisor

    return np.array(mat).flatten()


def singlet_transitions(sequence):
    to_index = {}
    seq_len = len(sequence)
    bases_len = len(bases)

    for i in range(bases_len):
        to_index[bases[i]] = i

    mat = np.zeros((bases_len, bases_len))
    hits = np.zeros((mat.shape[0],))
    a = sequence[0]

    for i in range(seq_len - 1):
        b = sequence[i + 1]
        mat[to_index[a]][to_index[b]] += 1
        hits[to_index[a]] += 1
        a = b

    for i in range(mat.shape[0]):
        # The correct way
        # divisor = hits[i]
        # Totally wrong, but works better!
        divisor = seq_len
        if divisor == 0:
            continue
        for j in range(mat.shape[1]):
            mat[i][j] /= divisor

    return np.array(mat).flatten()


def baseToNumber(base):
    x = base
    if base not in bases:
        x = basesMap[base]
    return basesNumerical[x]


def empty_singles():
    d = {}

    for i in bases:
        d[i] = 0.0
    return d


def singles_hist(sequence):
    singles = empty_singles()
    l = len(sequence)

    for i in range(l):
        char = sequence[i]

        if char not in singles:
            print("Unknown single-gram:", char)
        else:
            singles[char] += 1

    for key in singles:
        singles[key] /= l

    return np.asarray(list(singles.values()))


def emptyPairs():
    d = {}

    for i in bases:
        for j in bases:
            d[i + j] = 0.0
    return d


def pairHist(sequence):
    pairs = emptyPairs()
    l = len(sequence)
    a = sequence[0]

    for i in range(l - 1):
        b = sequence[i + 1]

        d = a + b
        if d not in pairs:
            print("Unknown digram:", d)
        else:
            pairs[d] += 1
        a = b

    for key in pairs:
        pairs[key] /= l - 1

    return np.asarray(list(pairs.values()))


def emptyTriplets():
    d = {}
    for i in bases:
        for j in bases:
            for k in bases:
                d[i + j + k] = 0.0
    return d


def tripletsHist(sequence):
    triplets = emptyTriplets()
    l = len(sequence)
    a = sequence[0]
    b = sequence[1]

    for i in range(l - 2):
        c = sequence[i + 2]

        d = a + b + c
        if d not in triplets:
            print("Unknown trigram:", d)
        else:
            triplets[d] += 1
        a = b
        b = c

    for key in triplets:
        triplets[key] /= l - 2

    return np.asarray(list(triplets.values()))


def wavelet(sequence, n=40):
    seq = np.array(list(map(baseToNumber, sequence)))
    a, b = pywt.dwt(seq, 'rbio1.3')
    freq_slice_a = a[:n]
    freq_slice_b = b[:n]
    # plt.plot(a)
    # plt.show()
    # exit()
    return np.append(freq_slice_a, freq_slice_b)


def fourier(sequence, n=100):
    seq = np.array(list(map(baseToNumber, sequence)))
    freq = np.fft.rfft(seq)
    freqSlice = freq[:n]  # freq[-n:]
    # concatenate real and imaginary parts to single vector
    return np.append(freqSlice[:].real, freqSlice[:].imag)


def oneHot(names):
    d = {}
    i = 0
    for name in names:
        code = np.zeros(len(names), dtype=np.float32)
        code[i] = 1.0
        d[name] = code
        i += 1
    return d


def integerCode(names):
    d = {}
    i = 0
    for name in names:
        d[name] = i  # np.array([i],dtype = np.float32)
        i += 1
    return d


dataDir = "./data"


def prepareData(featureExtractor, save=True, fastaName='current_Fungi_unaligned.fa', numSpecies=40):
    data = []
    labels = []
    labelsOneHot = []

    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    with open(fastaName) as fp:
        namesSeqMap = fastaParser.getNamesSeqSet(fp)
        print ("Parsed:", len(namesSeqMap), "species")

        sortedBySeqCount = sorted(namesSeqMap.items(), key=lambda x: len(x[1]), reverse=True)
        # sortedBySeqCount = filter(lambda x: "uncultured" not in x[0], sortedBySeqCount)

        print 'Taking', numSpecies, 'species with most sequences:'
        print(list([pair[0], len(pair[1])] for pair in sortedBySeqCount[0:numSpecies]))
        print 'Working with',\
            np.sum(list(len(pair[1]) for pair in sortedBySeqCount[0:numSpecies])), 'sequences in total'

        # codes = None
        # if oneHot:
        #    codes = oneHot(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))
        # else:
        #    codes = integerCode(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))

        intCodes = integerCode(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))
        oneHotCodes = oneHot(list(pair[0] for pair in sortedBySeqCount[0:numSpecies]))

        i = 0
        for pair in sortedBySeqCount:
            for seq in pair[1]:
                data.append(featureExtractor(seq))
                labels.append(intCodes[pair[0]])
                labelsOneHot.append(oneHotCodes[pair[0]])
            i += 1
            if i == numSpecies:
                break

    d = np.asarray(data)
    l = np.asarray(labels)
    loh = np.asarray(labelsOneHot)
    if save:
        np.save(dataDir + "/x.npy", d)
        np.save(dataDir + "/y.npy", l)
        np.save(dataDir + "/yoh.npy", loh)
    return d, l, loh


def loadData():
    data = np.load(dataDir + "/x.npy")
    labels = np.load(dataDir + "/y.npy")
    labelsOneHot = np.load(dataDir + "/yoh.npy")
    return data, labels, labelsOneHot
