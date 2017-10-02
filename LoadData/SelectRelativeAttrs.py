from LoadData.UserRecord import *
import math


def getColume(colIndex,data):
    #get a colume of data at colIndex
    col=[]
    for r in data:
        col.append(r[colIndex])
    return col

def mean(col):
    #cal mean
    sum=0.0
    n=0
    for r in col:
        if r is not None:
            sum=sum+r
            n=n+1
    if n==0:
        return 0.0
    return sum/n

def variance(col,avg):
    #cal variance
    sumx2=0.0
    n=0
    for r in col:
        if r is not None:
            sumx2=sumx2+r*r
            n=n+1
    if n==0:
        sumx2=0.0
    else:
        sumx2=sumx2/n
    return math.sqrt(sumx2-avg*avg)

def nullCounter(col,n):
    #count None number
    count=0
    for r in col:
        if r is None:
            #print("None",r)
            count=count+1
    return count

def sparseCols(data,sparseRate=0.3):
    #given a sparseRate,filter those data
    indexList=[]
    attrdim=len(data[0])
    n=len(data)
    for i in range(attrdim):
        col=getColume(i,data)
        nullsum=nullCounter(col,n)
        if n*sparseRate<nullsum:
            #print(n,nullsum,i)
            indexList.append(i)
    return  indexList

def rmCols(data,indexL):
    #remove cols in the indexL
    newdata = []
    for record in data:
        a = []
        for i in range(len(record)):
            if i not in indexL:
                a.append(record[i])
        newdata.append(a)
    return newdata
def removeSparseCols(data,sparseIndex):
    #legacy support
    return rmCols(data,sparseIndex)

def fillSparseCols(data,sparseIndex):
    #using the predicting methods to fill in the missing value of a record
    pass

def valueTypeRcheck(col1,col2):
    #Pearson check
    #return a value r which lies between -1 and 1
    #0 means unrelated,1 is positive relation
    #-1 is negative relative
    n=len(col1)
    avg1=mean(col1)
    avg2=mean(col2)
    var1=variance(col1,avg1)
    var2=variance(col2,avg2)
    #print('mean1,mean2,var1,var2',avg1,avg2,var1,var2 )
    ys=0.0
    for i in range(n):
        if col1[i] is not None and col2[i] is not None:
            ys=ys+col1[i]*col2[i]
    if var1==0.0 or var2==0.0:
        #constant value ignored
        return 0.0
    r12=(ys-n*avg1*avg2)/(n*var1*var2)
    return r12

def Rlist(data,minR=0.02,maxR=0.5):
    #select the most related attributes
    rmL=[]
    relation=[]
    #rm those that unrelated to the list
    indexLable=len(data[0])-1
    label = getColume(indexLable-1, data)
    col=[]
    col1=[]
    col2=[]
    for i in range(indexLable):
        col=getColume(i,data)
        tag=True
        for _ in col:
            if _!=None:
                if type(_)==str:
                    tag=False
                break
        if tag==False:
            continue

        r=valueTypeRcheck(col,label)
        relation.append([i,indexLable,abs(r)])
    #relation=mergeSort2(relation)
    for _ in relation:
        if _[2]<minR:
            rmL.append(_[0])
    #between attrs,rm those unrelated
    for i in range(indexLable):
        if i in rmL:
            continue
        for j in range(indexLable):
            if j in rmL or j==i:
                continue
            #attach those attr
            relation=[]
            col1 = getColume(i, data)
            col2=getColume(j,data)
            tag = True
            for _ in col:
                if _ != None:
                    if type(_) == str:
                        tag = False
                    break
            if tag == False:
                continue

            r = valueTypeRcheck(col, label)
            relation.append([i, indexLable, abs(r)])
        #relation=mergeSort2(relation)
        #rm those attr that high related to another and owns lots of None
        for _ in relation:
            if _[2]>maxR:
                nc1=nullCounter(getColume(_[0],data),len(data))
                nc2=nullCounter(getColume(_[1],data),len(data))
                if nc1<=nc2:
                    rmL.append(_[0])
                else:
                    rmL.append(_[1])
    print(rmL)
    return rmL
#test
def main():
    mydata=UserTrainData('../data/train.csv')
    data=mydata.nextBatch()
    sparseIndex=sparseCols(data)
    print('SparseIndex(%d)'%(len(sparseIndex)),sparseIndex)
    data=removeSparseCols(data,sparseIndex)
    Rlist(data)
    '''
    for i in range(len(data[0])-1):
        col1=getColume(i,data)
        tag=True
        for _ in col1:
            if _!=None:
                if type(_)==str:
                    tag=False
                break
        if tag==False:
            continue
        col2=getColume(len(data[0])-1,data)

        print('(%d,%d)'%(i,len(data[0])-1),valueTypeRcheck(col1,col2))
    '''
if __name__=="__main__":
    main()
