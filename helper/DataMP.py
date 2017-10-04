def nullCounter(col):
    #count None number
    count=0
    for r in col:
        if r is None:
            #print("None",r)
            count=count+1
    return count
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
def nullCounter2(index,data):
    count=0
    for x in data:
        if x[index] is None:
            count=count+1
    return count
def sparseCols(data,sparseRate=0.3):
    #given a sparseRate,filter those data
    indexList=[]
    attrdim=len(data[0])
    n=len(data)
    for i in range(attrdim):
        col=getColume(i,data)
        nullsum=nullCounter(col)
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