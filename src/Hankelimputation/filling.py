from .functions1d import *
from .functions2d import *
import pandas as pd

def testmax_L(data):
    numdat = data.to_numpy()
    N,dim = data.shape
    pnt = int(N/2.8)
    print(pnt)
    for v in range(pnt):
        L = pnt - v
        try:
            print(L)
            if dim > 1:
                construct2d(numdat,L,dim)
            else:
                construct1d(numdat,L)
        except ValueError:
            if L % 5 == 0:
                print("less than",L)
            else:
                pass
        else:
            print(L)
            return L

def batchfilling(data,dname,bsize):
    numdat = data.to_numpy()
    bina = data.notna()
    L = testmax_L(data.iloc[:bsize])
    list_frm = [] 
    for batch in range(0,len(data)-bsize,bsize):
        print(batch,batch+bsize)
        filled = mcwb2d(numdat[batch:batch+bsize],L,0.1,bina[batch:batch+bsize],50000,3)
        frm = pd.DataFrame(filled)
        frm.to_csv(dname + "_filled1.csv",index=False)
        list_frm.append(frm)
    togat = pd.concat(list_frm)
    togat.to_csv(dname + "_filled.csv",index=False)
    return list_frm

def fullfilling(data,dname): 
    L = testmax_L(data) 
    #^prevent ValueError: All the input dimensions except for axis 0 must match exactly. cp.bmat(hank)
    numdat = data.to_numpy()
    bina = data.notna()
    filled = mcwb2d(numdat,L,0.1,bina,50000,3)
    frms = pd.DataFrame(filled)
    frms.to_csv(dname + "_filled.csv",index=False)
    return filled