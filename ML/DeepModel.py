from LoadData.UserRecord import *
import tensorflow as tf
import numpy as np
import time
import xml.dom.minidom as xmlparser
from LoadData.SelectRelativeAttrs import *

#load net structure from file
def readnetStructure(inDim,outDim):
    print("netWorks Structure")
    w = {}
    b = {}
    # READ PARAMETERS FROM ANNStructure.XML
    dom = xmlparser.parse("..\data\ANNStructure.xml")
    ANN = dom.documentElement
    prevNum=inDim
    nextNum=0
    layer=''
    i=0
    while True:
        try:
            layer='L'+str(i)
            print(layer)
            nextNum = eval(ANN.getElementsByTagName(layer)[0].childNodes[0].nodeValue)
            w[i]=[prevNum,nextNum]
            b[i]=[nextNum]
            prevNum=nextNum
            i = i + 1
        except Exception as e:
            w[i]=[prevNum,outDim]
            b[i]=[outDim]
            break

    return (w,b)
def getRunTuple(dataBatch,data):
    X=data.pureAttrs(dataBatch)
    Y=data.labelTransform(dataBatch)
    return (X,Y)
#deep leanring scheme
def multi_perceptron(trainData,validData,testData):
    #extract validdata
    xt,yt=getRunTuple(validData.nextBatch(),validData)
#def Learning parameters
    learnRate = 0.1
    batchSize = 500
    iterationNum = 1000
    inDim=len(trainData.dataSet[0])
    outDim=1
#define the Graph
    x=tf.placeholder(tf.float32,[None,inDim])
    y=tf.placeholder(tf.float32,[None,outDim])
    #drop out probability
    keep_prob=tf.placeholder(tf.float32)
    #def hidden layers
    w,b=readnetStructure(inDim,outDim)#defNetWork(inDim,outDim,hiddenLayer)

    W={}
    B={}
    H={}
    for i in range(len(b)):
        #print(i)
        print(w[i])
        print(b[i])
        W[i]=tf.Variable(tf.truncated_normal(shape=w[i],stddev=0.1))
        B[i]= tf.Variable(tf.constant(value=0.1, shape=b[i]))

    H[0]=tf.nn.sigmoid(tf.matmul(x,W[0])+B[0])
    for i in range(1,len(b)-1):
        #print(i)
        H[i]=tf.nn.sigmoid(tf.matmul(H[i-1],W[i])+B[i])

    model=tf.nn.sigmoid(tf.matmul(H[len(b)-2],W[len(b)-1])+B[len(b)-1])
    model=tf.nn.dropout(model,keep_prob)
    #error define
    #errorLayer=tf.Variable(tf.constant([model,1/(model+0.001)]),trainable=False)
    errorLayer=model*tf.slice(y,[0,0],[-1,1])+tf.slice(y,[0,1],[-1,1])/(model+0.001)
    loss=tf.reduce_sum(tf.square(errorLayer))
    train_step=tf.train.GradientDescentOptimizer(learning_rate=learnRate).minimize(loss)
    #test Graph

    predict=(tf.abs(model-tf.cast(tf.arg_max(y,1),tf.float32))<0.01)
    correctPrediction=tf.reduce_sum(tf.cast(predict,tf.float32))


    #begin train
    print("init variables")

    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    print("running multi-layer perceptrons")
    t1=time.time()
    prevtestAcc=0.0
    stableCounter=0#count check times for stable status
    maxCheck=3#def max check time for stable status
    for i in range(iterationNum):
        batchX,batchY=getRunTuple(trainData.nextBatch(batchSize),trainData)
        sess.run(train_step,feed_dict={x:batchX,y:batchY,keep_prob:0.5})
        if i % 20 == 0:
            print("step %d" % (i + 1))
            acctest=sess.run(correctPrediction,feed_dict={x:xt,y:yt,keep_prob:1.0})/validData.dataSize
            print("test accuracy=%f"%(acctest))
            acctrain = sess.run(correctPrediction, feed_dict={x: batchX, y:batchY,keep_prob: 1.0}) / len(batchX)
            print("train accuracy=%f" % (acctrain))
            if abs(prevtestAcc-acctest)<0.01:
                stableCounter = stableCounter + 1
                if stableCounter>maxCheck:
                    break
            else:
                prevtestAcc=acctest

    t2=time.time()
    print("finished in",t2-t1,"s")

    correctNum=sess.run(correctPrediction,feed_dict={x:xt,y:yt,keep_prob:1.0})

    print("accuracy is as below with learning rate={} and batchSize={}:".format(learnRate,batchSize))

    print(correctNum,validData.dataSize,correctNum/validData.dataSize)

    #predict result


#test
def main():
    # testdata
    testData = UserTestData('../data/test.csv')
    #model train data
    mydata=UserTrainData('../data/train.csv')
    #preprocessing
    rmL=loadRlist()
    mydata.dataSet=rmCols(mydata.dataSet,rmL)
    testData.testData=rmCols(testData.testData,rmL)
    #split 0.8 0.2
    ratio=0.8
    n=int(len(mydata.dataSet)*ratio)
    tdata=mydata.dataSet[n:]
    mydata.dataSize=n
    mydata.dataSet=mydata.dataSet[0:n]
    traindata=mydata
    validdata=UserTrainData(tdata)
    #begin
    print('data prepared')
    multi_perceptron(traindata,validdata,testData)


if __name__=="__main__":
    main()