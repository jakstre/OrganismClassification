import tensorflow as tf
import numpy as np
import os

optDir = "./opts"

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

       
        self.biases = tf.Variable(tf.zeros([1])) #tf.Variable(tf.zeros([nUnits]))
        

        self.params = [self.weights, self.biases]

    def compute(self,x):
        return tf.add(tf.matmul(x, self.weights), self.biases)

    def getParams(self):
        return self.params


class MLP(Link):
    def __init__(self, layerSizes):
        super(MLP, self).__init__()
        
        #self.nHidden = nHidden
        #self.nOut = nOut
        #self.nInput = nInput
        
        # tf Graph input/output
        self.input = tf.placeholder("float", [None, layerSizes[0]])
        self.target = tf.placeholder("float", [None, layerSizes[-1]])
        self.keepProb = tf.placeholder(tf.float32)

        self.layers = []
        self.params = []
        
        for i in range(len(layerSizes)-1):
            self.layers.append(Linear(layerSizes[i], layerSizes[i+1]))
            #print("creating linear", layerSizes[i], layerSizes[i+1])
            self.params.extend(self.layers[i].getParams())
        
        self.output = self.compute(self.input)
        self.context = None

    def compute(self,x):
        h = x
        for l in self.layers[:-1]:
            h = tf.nn.dropout(tf.nn.elu(l(h)), self.keepProb) 
        h = self.layers[-1](h)
        return h
        
    def getParams(self):
        return self.params

    # return best recorded score on test set
    def fit(self, xTrain,yTrain,xTest,yTest, verbose = True, save = True, batchSize = 200, numEpochs=1000, alpha= 0.0001, dropoutRatio = 0.1):

        if self.context == None:
            self.context = tfContext(self,alpha)

        if not os.path.exists(optDir):
            os.makedirs(optDir)

        batchX = np.zeros([batchSize, xTrain.shape[1]],dtype = np.float32)
        batchY = np.zeros([batchSize, yTrain.shape[1]],dtype = np.float32)

        bestAccuracy = 0.0
        numWithoutImproving = 0
        with tf.Session() as sess:
            sess.run(self.context.init)
            for i in range(numEpochs):
                for _ in range(xTrain.shape[0]//batchSize):
                    for j in range(batchSize):
                        # TODO shuffling would be perhaps better
                        index  = np.random.randint(xTrain.shape[0])
                        batchX[j] = xTrain[index]
                        batchY[j] = yTrain[index] 

                    _, c= sess.run([self.context.optimizer, self.context.loss], feed_dict={self.input: batchX,self.target: batchY , self.keepProb: 1-dropoutRatio})
                    #print(c/batchSize)

                valAccuracy = self.context.accuracy.eval({self.input:xTest, self.target:yTest, self.keepProb: 1.0})
                if verbose:
                    print("Epoch:", i, "validation accuracy:", valAccuracy)
                if valAccuracy>bestAccuracy:
                    bestAccuracy = valAccuracy
                    numWithoutImproving = 0
                    if save:
                        _ = self.context.saver.save(sess, optDir + "/best.ckpt")
                else:
                    numWithoutImproving+=1

                if numWithoutImproving > 40:
                    break
        if verbose:
            print ("Done, best accuracy:",bestAccuracy)
        return bestAccuracy

    def test(self,xTest,yTest, load = False):
    
        with tf.Session() as sess:
            if load:
                self.context.saver.restore(sess, optDir + "/best.ckpt")
            acc = self.context.accuracy.eval({self.input:xTest, self.target:yTest, self.keepProb: 1.0})
            #print("Accuracy:", accuracy)

        return acc

    # TODO dont start new session each time
    #def load(self, file = optDir + "/best.ckpt"):
    #    with tf.Session() as sess:
    #        self.context.saver.restore(sess, file)

class tfContext(object):
    def __init__(self,net,alpha):
        super(tfContext, self).__init__()
        self.loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = net.output, labels = net.target))
        #optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum =0.8).minimize(loss)
        self.optimizer =tf.train.AdamOptimizer(learning_rate = alpha).minimize(self.loss)
        #optimizer =tf.train.AdagradOptimizer(learning_rate = alpha).minimize(loss) 
        #optimizer =tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(loss)
        self.correctPrediction = tf.equal(tf.argmax(net.output, 1), tf.argmax(net.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, "float"))
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        tf.get_default_graph().finalize()

    