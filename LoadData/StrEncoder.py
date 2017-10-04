from LoadData.SelectRelativeAttrs import *

def PresenceCounter(col):
    counter={}

    for x in col:
        #print(type(x))
        if x=='':
            continue
        ss = x.split('@')
        # print(ss)
        for s in ss:
            if s == '':
                continue
            if s in counter.keys():
                counter[s] = counter[s] + 1
            else:
                counter[s] = 1
    return counter
def rmLowSupport(counter,rate=1.0,maxKeys=50):
    vsum=0
    #print('again')
    for k in counter.keys():
        vsum=vsum+counter[k]
    avg=vsum/len(counter.keys())*rate
    rmK=set()
    for k in counter.keys():
        if counter[k]<avg:
            rmK.add(k)
    for e in rmK:
        del counter[e]
    if len(counter.keys())>maxKeys:
        rmLowSupport(counter)
def sumCounterValues(counter):
    sum=0
    for k in counter.keys():
        sum=sum+counter[k]
    return sum
def transferCounters(counters):

    for key in counters.keys():
        counter=counters[key]
        n=sumCounterValues(counter)
        for k in counter.keys():
            counter[k]=counter[k]/float(n)
    return counters
def setValuesStr(counter,index,data):
    for x in data:
        if '@' in x[index]:
            x[index]=len(x[index].split('@'))
            continue
        if x[index] in counter.keys():
            x[index]=counter[x[index]]
        else:
            x[index]=0.0
def StrEncode(counters,data):
    transferCounters(counters)
    for i in counters:
        setValuesStr(counters[i],i,data)
def getCounters(mydata):
    counters = {}
    data=mydata.dataSet
    for sAttr in mydata.TrainDataIgnoreStrIndex:
        col = getColume(sAttr, data)
        d_sta = PresenceCounter(col)

        #print('col=%d' % (sAttr), len(d_sta.keys()))
        rmLowSupport(d_sta)
        #print('col=%d' % (sAttr), len(d_sta.keys()))
        counters[sAttr] = d_sta
    return counters
#test
def main():
    mydata=UserTrainData('../data/train.csv')
    counters=getCounters(mydata)

    StrEncode(counters,mydata.dataSet)
    for i in range(120):
        print(mydata.dataSet[i])
if __name__=='__main__':
    main()

