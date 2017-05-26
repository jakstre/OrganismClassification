import tensorflow as tf
import numpy as np

class Link(object):
    def __init__(self):
        super(Link, self).__init__()

    def __call__(self,x):
        return self.compute(x)

    def compute(self,x):
        pass

    def getParams(self):
        pass

class Linear(Link):
    def __init__(self,nInput,nUnits):
        super(Linear, self).__init__()
        # Store layers weight & bias
        # init ~ 1/sqrt(n) * random sample from normal distribution
        # zero biases
        self.weights =  tf.Variable(tf.scalar_mul(tf.sqrt(1./float(nInput)),
                                            tf.random_normal([nInput, nUnits])))

       
        self.biases = tf.Variable(tf.zeros([1]))
        

        self.params = [self.weights, self.biases]

    def compute(self,x):
        return tf.add(tf.matmul(x, self.weights), self.biases)

    def getParams(self):
        return self.params


class MLP(Link):
    def __init__(self,nInput,nHidden,nOut):
        super(MLP, self).__init__()
        self.nHidden = nHidden
        self.nOut = nOut
        self.nInput = nInput
        
        # tf Graph input/output
        self.input = tf.placeholder("float", [None, nInput])
        self.target = tf.placeholder("float", [None, nOut])
        self.layer1 = Linear(nInput,nHidden)
        self.layer2= Linear(nHidden,nOut)

        p = self.layer1.getParams()
        self.params = p.extend(self.layer2.getParams())

    def compute(self,x):
        a1 = tf.nn.relu(self.layer1(x))
        return self.layer2(a1)

    def getParams(self):
        return self.params

