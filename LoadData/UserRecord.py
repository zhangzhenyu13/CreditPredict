import csv
class Data:
    TestDataStrIndex=[18,19,20,21,22,23,24,44,123,124,125]
    TrainDataStrIndex=[18,19,20,21,22,23,24,44,123,124,125]
    def __init__(self,file=None):
        '''
        the base is for settings
        load data from csv file
        if file is None, init a null dataSet
        '''

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
                try:
                    a=record[i]
                    x=eval(a)
                    record[i]=x
                except:
                    if train:
                        if i not in self.TrainDataStrIndex:
                            record[i]=None
                    else:
                        if i not in self.TestDataStrIndex:
                            record[i]=None
                    pass

        return  data
class UserTrainData(Data):
    dataSet=None
    dataSize=0
    __cur=0
    def __init__(self,file=None):
        print("Train Data")
        Data.__init__(file)
        if file is None:
            print("dataSize=%d,file=%s" % (0, file))
            pass
        else:
            f=open(file,"r")
            #print(f.read())
            reader=csv.reader(f)
            self.dataSet=[row[1:] for row in reader]
            del self.dataSet[0]
            self.dataSize=len(self.dataSet)

            print("dataSize=%d,file=%s"%(self.dataSize,file))
            f.close()
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
            data=data+self.dataSize[0:num_left]
            self.__cur=num_left
        return data
class UserTestData(Data):
    testData=None
    testSize=0
    def __init__(self,file):
        print("Test data")
        Data.__init__(file)
        if file is None:
            print("dataSize=%d,file=%s" % (0, file))
            pass
        else:
            f=open(file,"r")
            #print(f.read())
            reader=csv.reader(f)
            self.testData=[row[1:] for row in reader]
            del self.testData[0]
            self.testSize=len(self.testData)
            print("dataSize=%d,file=%s"%(self.testSize,file))
            f.close()
    def getData(self):
        return self.testData
def main():
    data=UserTrainData('../data/train.csv')

    for i in range(10):
        print(data.nextBatch(1))

    test = UserTestData('../data/test.csv')
    testdata=test.getData()
    testdata=test.parseNumAtr(testdata,train=False)
    for i in range(10):
        print(testdata[i])
if __name__=="__main__":
    main()