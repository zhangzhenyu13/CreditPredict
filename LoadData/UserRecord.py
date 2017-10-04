'''
this module is aimed to provide better data management
use Data Object to manage data
use labelTransform to get useful labels which are served as Y in train phase
use puerAttrs to serve as X in train phase
'''
import csv
from xml.dom.minidom import parse
import xml.dom.minidom

class Data(object):
    usefulAttr=0
    TestDataIgnoreStrIndex=[]
    TrainDataIgnoreStrIndex=[]
    def __init__(self,file=None):
        '''
        the base is for settings
        load data from csv file
        if file is None, init a null dataSet
        '''
        # READ PARAMETERS FROM DataConfig.XML
        #Ignore Attibute Index
        DOMTree = xml.dom.minidom.parse("../data/DataConfig.xml")
        config = DOMTree.documentElement
        testIndex = config.getElementsByTagName("TestDataIgnoreStrIndex")[0].firstChild.data
        trainIndex = config.getElementsByTagName("TrainDataIgnoreStrIndex")[0].firstChild.data
        testIndex=testIndex.split(',')
        trainIndex=trainIndex.split(',')
        self.TestDataIgnoreStrIndex=[eval(a) for a in testIndex]
        self.TrainDataIgnoreStrIndex=[eval(a) for a in trainIndex]
        #print(self.TestDataIgnoreStrIndex,self.TrainDataIgnoreStrIndex)
        #
    def nextBatch(self,dataSize=None):
        '''
        if dataSize is None, then return all the data
        else return a dataSet of size dataSize
        :param dataSize:
        :return: a list of user reocord
        '''
    def parseNumAtr(self,data,train=True):
        i=0
        for record in data:
            for i in  range(len(record)):
                if train:
                    if i  in self.TrainDataIgnoreStrIndex:
                        continue
                else:
                    if i in self.TestDataIgnoreStrIndex:
                        continue
                try:
                    a=record[i]
                    x=eval(a)
                    record[i]=x
                except:
                    if train:
                        if i not in self.TrainDataIgnoreStrIndex:
                            record[i]=None
                    else:
                        if i not in self.TestDataIgnoreStrIndex:
                            record[i]=None
                    pass

        return  data
    def labelTransform(self,data):
        #this function is aimed at transform the label,which is 0/1, to be a 2-dim vector
        #if label of data is 1,which means bad, then label vector will be [0,1]
        #when it's 0,which means good, then label vector will be [1,0]
        #the return can be regarded as Y
        labels=[]
        for record in data:
            act=record[len(record)-1]
            if act==0:
                labels.append([1,0])
            else:
                labels.append([0,1])
        return labels
    def pureAttrs(self,dataSet):
        #delete the target attibute of each record so that the return is served as X
        data=[]
        index=len(dataSet[0])-1
        for record in dataSet:
            data.append(record[0:index])
        return data
    def getRunTuple(self,dataBatch):
        Y = self.labelTransform(dataBatch)
        X = self.pureAttrs(dataBatch)
        return (X, Y)
class UserTrainData(Data):
    dataSet=None
    dataSize=0
    __cur=0
    __curXY=0
    X=None
    Y=None
    def __init__(self,file=None):
        print("Train Data")
        Data.__init__(self,file)
        if file is None:
            print("dataSize=%d,file=%s" % (0, file))
            return
        elif type(file)==list:
            self.dataSet = file
            self.dataSize = len(file)
        else:
            f=open(file,"r")
            #print(f.read())
            reader=csv.reader(f)
            self.dataSet=[row[1:] for row in reader]
            del self.dataSet[0]
            self.dataSize=len(self.dataSet)
            self.dataSet=self.parseNumAtr(self.dataSet,True)
            print("dataSize=%d,file=%s"%(self.dataSize,file))
            f.close()
    def initXY(self):
        self.X, self.Y = self.getRunTuple(self.dataSet)
    def nextBatch(self,dataSize=None):
        if dataSize is None:
            return self.dataSet
        data=[]
        if self.__cur+dataSize<=self.dataSize:
            data= self.dataSet[self.__cur:self.__cur+dataSize]
            self.__cur=self.__cur+dataSize
            if self.__cur==self.dataSize:
                self.__cur=0
        else:
            num_left=dataSize-(self.dataSize-self.__cur)
            data=self.dataSet[self.__cur:]
            data=data+self.dataSet[0:num_left]
            self.__cur=num_left
        return data
    #next X,Y
    def nextXY(self,dataSize):
        x=[]
        y=[]
        if self.__curXY+dataSize<=self.dataSize:
            x= self.X[self.__curXY:self.__curXY+dataSize]
            y=self.Y[self.__curXY:self.__curXY+dataSize]
            self.__curXY=self.__curXY+dataSize
            if self.__curXY==self.dataSize:
                self.__curXY=0
        else:
            num_left=dataSize-(self.dataSize-self.__curXY)
            x=self.X[self.__curXY:]
            y=self.Y[self.__curXY:]
            x=x+self.X[0:num_left]
            y=y+self.Y[0:num_left]
            self.__curXY=num_left
        return (x,y)

class UserTestData(Data):
    dataSet=[]
    IDset=[]
    dataSize=0
    def __init__(self,file):
        print("Test data")
        super(UserTestData,self).__init__(file)
        if file is None:
            print("dataSize=%d,file=%s" % (0, file))
            pass
        else:
            f=open(file,"r")
            #print(f.read())
            reader=csv.reader(f)
            for row in reader:
                self.dataSet.append( row[1:])
                self.IDset.append(row[0])
            del self.dataSet[0]
            del self.IDset[0]
            self.dataSize=len(self.dataSet)
            self.dataSet=self.parseNumAtr(self.dataSet,False)
            print("dataSize=%d,file=%s"%(self.dataSize,file))
            f.close()
    def getData(self):
        return self.dataSet

#navie test for the correctness
def main():
    #test train data interface
    data=UserTrainData('../data/train.csv')
    traindata=data.nextBatch(10)
    #traindata=data.parseNumAtr(traindata,True)

    labels=data.labelTransform(traindata)

    for l in labels:
        print(l)
    for i in range(10):
        print(len(traindata[i]))
    traindata = data.pureAttrs(traindata)
    for i in range(10):
        print(len(traindata[i]))
    #test testdata interface
    test = UserTestData('../data/test.csv')
    testdata=test.getData()
    #testdata=test.parseNumAtr(testdata,train=False)
    for i in range(10):
        print(testdata[i])
if __name__=="__main__":
    main()
