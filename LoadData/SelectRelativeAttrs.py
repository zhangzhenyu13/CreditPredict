from LoadData.UserRecord import *
import math
import helper.personalizedSort

def getColume(colIndex,data):
    #get a colume of data at colIndex
    col=[]
    for r in data:
        col.append(r[colIndex])
    return col

def getColType(col):
    for _ in col:
        if _ != None:
            return type(_)
    return None

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
    if var1==0.0 or var2==0.0:
        #constant value ignored
        return 0.0
    ys=0.0
    for i in range(n):
        if col1[i] is not None and col2[i] is not None:
            ys=ys+col1[i]*col2[i]

    r12=(ys-n*avg1*avg2)/(n*var1*var2)
    return r12
def loadRlist():
    rmL=[]
    with open('rmList','r') as f:
        s=f.read()
        ss=s.split(',')
        for s in ss:
            try:
                i=eval(s)
                rmL.append(i)
            except:
                break
    return rmL
def Rlist(data,minR=0.01,maxR=0.95):
    #select the most related attributes
    rmL=set()
    relation=[]
    #rm those that unrelated to the list
    indexLable=len(data[0])-1
    label = getColume(indexLable-1, data)
    col=[]
    col1=[]
    col2=[]
    for i in range(indexLable):
        col=getColume(i,data)
        if getColType(col)==str:
            continue

        r=valueTypeRcheck(col,label)
        relation.append([i,indexLable,abs(r)])
    #relation=mergeSort2(relation)
    for _ in relation:
        if _[2]<minR:
            rmL.add(_[0])
    #print('unrelated with target',len(rmL),rmL)
    #between attrs,rm those unrelated
    for i in range(indexLable):
        if i in rmL:
            continue
        for j in range(i+1,indexLable):
            if j in rmL or j==i:
                continue
            #attach those attr
            relation=[]
            col1 = getColume(i, data)
            col2=getColume(j,data)

            #make sure type of col1 and col2
            if getColType(col1)==str or getColType(col2)==str:
                continue

            r = valueTypeRcheck(col1,col2)
            relation.append([i, j, abs(r)])

        #rm those attr that high related to another and owns lots of None
        relation=helper.personalizedSort.mergeSort2(relation)
        for _ in relation:
            if _[2]>maxR:
                nc1=nullCounter(getColume(_[0],data),len(data))
                nc2=nullCounter(getColume(_[1],data),len(data))
                if nc1>nc2:
                    rmL.add(_[0])
                    break
                else:
                    rmL.add(_[1])
        #print('iter %d'%(i),len(rmL),rmL)
    rmL=list(rmL)
    rmL=helper.personalizedSort.quickSort2(rmL)
    #print(len(rmL),rmL)
    with open("rmList","w") as f:
        for i in rmL:
            f.write(str(i)+",")
        f.close()
    return rmL
#test
def main():
    mydata=UserTrainData('../data/train.csv')
    data=mydata.nextBatch(100)
    sparseIndex=sparseCols(data)
    print('SparseIndex(%d)'%(len(sparseIndex)),sparseIndex)
    data=removeSparseCols(data,sparseIndex)
    rmL=Rlist(data)
    rmCols(data,rmL)
    rmL=loadRlist()
    print(rmL)
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
