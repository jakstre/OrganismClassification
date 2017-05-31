import tensorflow as tf
import numpy as np
import os

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
    def __init__(self, layerSizes):
        super(MLP, self).__init__()
        
        #self.nHidden = nHidden
        #self.nOut = nOut
        #self.nInput = nInput
        
        # tf Graph input/output
        self.input = tf.placeholder("float", [None, layerSizes[0]])
        self.target = tf.placeholder("float", [None, layerSizes[-1]])

        self.layers = []
        self.params = []
        
        for i in range(len(layerSizes)-1):
            self.layers.append(Linear(layerSizes[i], layerSizes[i+1]))
            #print("creating linear", layerSizes[i], layerSizes[i+1])
            self.params.extend(self.layers[i].getParams())
        
        self.output = self.compute(self.input)

    def compute(self,x):
        h = x
        for l in self.layers[:-1]:
            h = tf.nn.relu(l(h))

        h = self.layers[-1](h)
        return h
        
        #a1 = tf.nn.relu(self.layers[0](x))
        #return self.layers[1](a1)
        #a1 = tf.nn.relu(self.layer1(x))
        #return self.layer2(a1)

    def getParams(self):
        return self.params







optDir = "./opts"

def NNfit(net,xTrain,yTrain,xTest,yTest, batchSize = 64, numEpochs=1000, alpha= 0.0001):

    if not os.path.exists(optDir):
        os.makedirs(optDir)

    batchX = np.zeros([batchSize, xTrain.shape[1]],dtype = np.float32)
    batchY = np.zeros([batchSize, yTrain.shape[1]],dtype = np.float32)

    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = net.output, labels = net.target))
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum =0.8).minimize(loss)
    #optimizer =tf.train.AdamOptimizer(0.1).minimize(loss)

    correctPrediction = tf.equal(tf.argmax(net.output, 1), tf.argmax(net.target, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))

    bestAccuracy = 0.0
    numWithoutImproving = 0

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    #tf.get_default_graph().finalize() 
    

    with tf.Session() as sess:
        sess.run(init)
        valAccuracy = accuracy.eval({net.input:xTest, net.target:yTest})
        print("Epoch:", -1, "validation accuracy:", valAccuracy)
        for i in range(numEpochs):
            for _ in range(xTrain.shape[0]//batchSize):
                for j in range(batchSize):
                    # TODO shuffling would be perhaps better
                    index  = np.random.randint(xTrain.shape[0])
                    batchX[j] = xTrain[index]
                    batchY[j] = yTrain[index] 

                _, c= sess.run([optimizer, loss], feed_dict={net.input: batchX,net.target: batchY})
                #print(c/batchSize)

            valAccuracy = accuracy.eval({net.input:xTest, net.target:yTest})
            print("Epoch:", i, "validation accuracy:", valAccuracy)
            if valAccuracy>bestAccuracy:
                bestAccuracy = valAccuracy
                numWithoutImproving = 0
                _ = saver.save(sess, optDir + "/best.ckpt")
            else:
                numWithoutImproving+=1

            if numWithoutImproving > 25:
                break
 
    return

def NNtest(net,xTest,yTest):
    
    correctPrediction = tf.equal(tf.argmax(net.output, 1), tf.argmax(net.target, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, optDir + "/best.ckpt")
        acc=  accuracy.eval({net.input:xTest, net.target:yTest})
        #print("Accuracy:", accuracy)

    return acc

    


