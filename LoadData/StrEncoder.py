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
def rmLowSupport(counter,rate=1.0,maxKeys=20):
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
#test
def main():
    mydata=UserTrainData('../data/train.csv')
    data=mydata.nextBatch()
    for sAttr in mydata.TrainDataIgnoreStrIndex:
        col=getColume(sAttr,data)
        d_sta=PresenceCounter(col)

        print('col=%d'%(sAttr),len(d_sta.keys()))
        rmLowSupport(d_sta)
        print('col=%d' % (sAttr), len(d_sta.keys()))
if __name__=='__main__':
    main()

