from LoadData.UserRecord import *
import math
def getColume(colIndex,data):
    col=[]
    for r in data:
        col.append(r[colIndex])
    return col
def mean(col):
    sum=0.0
    n=0
    for r in col:
        if r is not None:
            sum=sum+r
            n=n+1
    return sum/n
def variance(col,avg):
    sumx2=0.0

    n=0
    for r in col:
        if r is not None:
            sumx2=sumx2+r*r
            n=n+1
    sumx2=sumx2/n
    return math.sqrt(sumx2-avg*avg)

def nullCounter(col,n):
    count=0
    for r in col:
        if r is None:
            #print("None",r)
            count=count+1
    return count
def sparseCols(data):
    indexList=[]
    attrdim=len(data[0])
    n=len(data)
    for i in range(attrdim):
        col=getColume(i,data)
        nullsum=nullCounter(col,n)
        if n*0.3<nullsum:
            #print(n,nullsum,i)
            indexList.append(i)
    return  indexList
def removeSparseCols(data,sparseIndex):
    newdata=[]
    for record in data:
        a=[]
        for i in range(len(record)):
            if i not  in sparseIndex:
                a.append(record[i])
        newdata.append(a)
    return newdata
def fillSparseCols(data,sparseIndex):
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
    print('mean1,mean2,var1,var2',avg1,avg2,var1,var2 )
    ys=0.0
    for i in range(n):
        if col1[i] is not None and col2[i] is not None:
            ys=ys+col1[i]*col2[i]
    r12=(ys-n*avg1*avg2)/(n*var1*var2)
    return r12
#test
def main():
    mydata=UserTrainData('../data/train.csv')
    data=mydata.nextBatch()
    sparseIndex=sparseCols(data)
    print('SparseIndex(%d)'%(len(sparseIndex)),sparseIndex)
    removeSparseCols(data,sparseIndex)
    for i in range(len(data[0])-1):
        if i in mydata.TrainDataIgnoreStrIndex:
            continue
        col1=getColume(i,data)
        col2=getColume(len(data[0])-1,data)

        print('(%d,%d)'%(i,len(data[0])-1),valueTypeRcheck(col1,col2))

if __name__=="__main__":
    main()
