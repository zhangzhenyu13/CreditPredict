#coding: utf-8
import tensorflow as tf
import numpy as np
import time
import xml.dom.minidom as xmlparser
from LoadData.SelectRelativeAttrs import *
from LoadData.StrEncoder import *
from LoadData.fillStrategy import *
import pandas as pd
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

            nextNum = eval(ANN.getElementsByTagName(layer)[0].childNodes[0].nodeValue)
            w[i]=[prevNum,nextNum]
            b[i]=[nextNum]
            prevNum=nextNum

            print(layer, ":", w[i], b[i])

            i = i + 1
        except Exception as e:
            w[i]=[prevNum,outDim]
            b[i]=[outDim]
            break

    return (w,b)

#deep leanring scheme
def multi_perceptron(trainData,validData,testData):
    #extract validdata
    xt,yt=validData.getRunTuple(validData.dataSet)
#def Learning parameters
    learnRate = 0.1
    batchSize = 100
    iterationNum = 1000
    inDim=len(xt[0])
    outDim=1
#define the Graph
    x=tf.placeholder(tf.float32,[None,inDim],name='X_in')
    y=tf.placeholder(tf.float32,[None,2],name='Y_out')
    #drop out probability
    keep_prob=tf.placeholder(tf.float32)
    #def hidden layers
    w,b=readnetStructure(inDim,outDim)#defNetWork(inDim,outDim,hiddenLayer)

    W={}
    B={}
    H={}
    for i in range(len(b)):

        W[i]=tf.Variable(tf.truncated_normal(shape=w[i],stddev=0.1))
        B[i]= tf.Variable(tf.constant(value=0.1, shape=b[i]))

    H[0]=tf.nn.sigmoid(tf.matmul((x),W[0])+B[0])
    for i in range(1,len(b)-1):
        #print(i)
        H[i]=tf.nn.sigmoid(tf.matmul(H[i-1],W[i])+B[i])

    model=tf.nn.sigmoid(tf.matmul(H[len(b)-2],W[len(b)-1])+B[len(b)-1])
    model=tf.nn.dropout(model,keep_prob)
#define error and train goal
    #errorLayer=tf.Variable(tf.constant([model,1/(model+0.001)]),trainable=False)
    errorLayer=model*tf.slice(y,[0,0],[-1,1])+tf.slice(y,[0,1],[-1,1])/(model+0.01)
    loss=tf.reduce_sum(tf.square(errorLayer))
    train_step=tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)
# correctness counter
    predict=(tf.abs(model-tf.cast(tf.slice(y,[0,1],[-1,1]),tf.float32))<0.1)
    correctPrediction=tf.reduce_sum(tf.cast(predict,tf.float32))
    accuracy=tf.reduce_mean(tf.cast(predict,tf.float32))
#begin train

    with tf.Session() as sess:
        print("init variables")
        init=tf.global_variables_initializer()
        #sess=tf.Session()
        sess.run(init)

        '''#/debug
        batchX, batchY = trainData.nextXY(10)
        print(sess.run(y,feed_dict={y:batchY}))
        print(sess.run(tf.slice(y,[0,0],[-1,1]),feed_dict={y:batchY}))
        print(sess.run(tf.slice(y,[0,1],[-1,1]),feed_dict={y:batchY}))
        #debug/'''

        prevtestAcc=0.0
        stableCounter=0#count check times for stable status
        maxCheck=1#def max check time for stable status
        #test before
        print("before train")
        sum=correctPrediction.eval(feed_dict={x: xt, y: yt,keep_prob:1.0})
        print("test accuracy=%f" % (sum))
        #begin train model
        print("running multi-layer perceptrons")
        t1 = time.time()
        for i in range(iterationNum):
            batchX,batchY=trainData.nextXY(batchSize)


            if i % 50 == 0:
                print("step %d" % (i))

                acctest=accuracy.eval(feed_dict={x:xt,y:yt,keep_prob:1.0})
                print("test accuracy=%2.2f"%(acctest))
                acctrain = accuracy.eval(feed_dict={x: batchX, y:batchY,keep_prob:1.0})
                print("train accuracy=%2.2f" % (acctrain))

                if abs(prevtestAcc-acctest)<0.01:
                    stableCounter = stableCounter + 1
                    if stableCounter>maxCheck:
                        break
    
                prevtestAcc=acctest

            sess.run(train_step, feed_dict={x: batchX, y: batchY, keep_prob: 0.5})
        t2=time.time()
        print("finished in",t2-t1,"s")

        correctNum=sess.run(correctPrediction,feed_dict={x:xt,y:yt,keep_prob:1.0})

        print("accuracy is as below with learning rate={} and batchSize={}:".format(learnRate,batchSize))

        print(correctNum,validData.dataSize,correctNum/validData.dataSize)

        #predict result
        pm=sess.run(model,feed_dict={x:xt,keep_prob:1.0})
        pt=sess.run(y,feed_dict={y:yt})
        for i in range(20):
            print(pm[i],pt[i])

        if testData is None:
            return
        result=sess.run(model,feed_dict={x:testData.dataSet,keep_prob:1.0})
    writePreiction(result,testData.IDset)
#write predicton result
def writePreiction(result,IDset):
    predict=[]
    #print(result)
    #print(len(result),len(IDset))
    for i in range(len(result)):
        predict.append([bytes(IDset[i],encoding='utf-8'),bytes("%2.2f"%(result[i][0]),encoding='utf-8')])
        #print(predict[i])
    f=open('../data/submission.csv','wb')
    writer = csv.writer(f)
    writer.writerow([bytes('id',encoding='utf-8'),bytes('predict',encoding='utf-8')])
    writer.writerows(predict)

#test
def main():
    # testdata
    testData = UserTestData('../data/test.csv')
    #model train data
    mydata=UserTrainData('../data/train.csv')
    #preprocessing
    StrEncode(getCounters(mydata),mydata.dataSet)
    StrEncode(getCounters(testData),testData.dataSet)
    rmL=loadRlist()
    mydata.dataSet=rmCols(mydata.dataSet,rmL)
    testData.dataSet=rmCols(testData.dataSet,rmL)
    fillColsMode(mydata.dataSet)
    fillColsMode(testData.dataSet)
    #split 0.8 0.2 for train model and 0.2 for valid correctness
    ratio=0.8
    n=int(len(mydata.dataSet)*ratio)
    tdata=mydata.dataSet[n:]
    mydata.dataSize=n
    mydata.dataSet=mydata.dataSet[0:n]
    traindata=mydata
    validdata=UserTrainData(tdata)
    traindata.initXY()
    validdata.initXY()
    #begin
    for i in range(20):
        print(traindata.dataSet[i])
    print('data prepared:Selected Attribute(%d), train Model(%d),validData(%d)'%
          (len(traindata.X[0]),len(traindata.X),len(validdata.X))
          )
    multi_perceptron(traindata,validdata,testData)


if __name__=="__main__":
    main()