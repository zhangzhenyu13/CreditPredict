def getMode(index,data):
    m_dict={}
    maxK=None
    for x in data:
        if x[index] is None:
            continue
        if x[index] in m_dict.keys():
            m_dict[x[index]]=m_dict[x[index]]+1
            if m_dict[x[index]]>maxK:
                maxK=x[index]
        else:
            m_dict[x[index]]=1
            if maxK is None:
                maxK=x[index]
    return maxK
def fillColsMode(data):
    #using the mode of the colume to fill in the missing value of a record
    mode={}
    n=len(data[0])
    for i in range(n):
        mode[i]=getMode(i,data)
    #filling
    for x in data:
        for i in range(n):
            if x[i] is None:
                x[i]=mode[i]