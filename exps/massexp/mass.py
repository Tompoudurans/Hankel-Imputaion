from math import pi
import hankelimputation
import random
import numpy
import pandas as pd
import scipy.optimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as mp
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import scipy

utils = importr('utils')
base = importr('base')
its = importr("imputeTS")
pandas2ri.activate()

def mkhle(data,percent):
    x, y = data.shape
    count = 0
    aim = round(x * y * percent)
    while count < aim:
        row = random.randint(1, x - 2)
        data.iloc[row, :] = numpy.nan
        count = data.isna().sum().sum()
        if count % 10 == 0:
            print(count, "(", aim, ")/", x * y, sep="")
    return data

def loopar(y,trm,non,sigma):
    print(trm)
    mu = 0
    for i in range(non):
        bas = random.gauss(mu, sigma)
        y.append(autor(y[i:i+len(trm)],trm) + bas)
    return numpy.array(y)

def autor(x,trm):
    y = 0
    for i in range(len(trm)):
        try:
            y = y + x[i]*trm[i]
        except IndexError:
            print("bad index")
        return y

class ewmsug:
    def __init__(self,data):
        self.data = data
        self.binar = data.isna()
        self.da = self.masker(self.data)
        
    def masker(self,data):
        da = data.copy()
        da[self.binar] = 0
        return da
    
    def ewmal(self,alpha):
        ewm = self.data.copy().ewm(alpha=alpha).mean()
        return mean_squared_error(self.masker(ewm),self.da)
    
    def optim(self,alpha):
        self.res = scipy.optimize.minimize(self.ewmal,
                       alpha,
                       method='nelder-mead',
                       options={'xatol': 1e-8, 'disp': True},
                       bounds=scipy.optimize.Bounds(0,1)
                      )
    
    def create_ewm(self):
        self.optim(0.5)
        return self.data.copy().ewm(alpha=self.res.x[0]).mean()

def fullana(filleds,org,withgap):
    emas = ewmsug(withgap)
    ewms = emas.create_ewm()
    interps = withgap.copy().interpolate()
    spline = its.na_interpolation(withgap.copy(),option = "spline")
    stine = its.na_interpolation(withgap.copy(),option = "stine")
    sea = its.na_seadec(withgap.copy())
    kal = its.na_kalman(withgap.copy())
    seap = its.na_seasplit(withgap.copy())
    datalist = [
        spline[0].reshape(-1,1),
        stine[0].reshape(-1,1),
        sea.reshape(-1,1),
        kal[0].reshape(-1,1),
        seap.reshape(-1,1),
        ewms.to_numpy(),
        interps.to_numpy(),
        filleds.to_numpy()
    ]
    table = []
    for i in range(len(datalist)):
        print(i,end="-")
        datalist[i] = maskorg(withgap,datalist[i])
        table.append(ana(org,datalist[i]))
    print("8")
    res = pd.DataFrame(table).transpose()
    res.columns = [
        "spline",
        "stine",
        "sea",
        "kal",
        "seap",
        "emas",
        "interps",
        "dualhi"
        ]
    return res



def maskorg(miss,fill):
    binpos = miss.isna()
    mix = miss.copy().to_numpy()
    mix[binpos] = fill[binpos]
    return mix

def fill(data):
    return hankelimputation.processing(data,0,0.1,max_iters=4500,verbose=True)

def create(trms,lengh,burn):
    data = []
    for j in range(len(trms)):
        y = [0]*len(trms[j])
        data.append(loopar(y.copy(),trms[j],lengh,0.1))
    frame = pd.DataFrame(data).transpose().iloc[burn:]
    return frame.reset_index(drop=True)

def mutiar(trm,val,sig):
    bas = numpy.random.normal(0,sig,len(trm))
    reg = numpy.matmul(trm,val)
    nex = reg + bas
    return nex

def mass_mutiar(trm,length,sig):
    x = len(trm)
    val = numpy.array([0]*x)
    for i in range(length):
        if i == 0:
            nex = mutiar(trm,val,sig)
            val = numpy.append([val],[nex],axis=0)
        else:
            nex = mutiar(trm,val[i],sig)
            val = numpy.append(val,[nex],axis=0)
    return val

def create2d(trms,lengh,burn):
    data = mass_mutiar(trms,lengh,0.1)
    frame = pd.DataFrame(data).iloc[burn:]
    return frame.reset_index(drop=True)

def pern(v,norm=True):
    w = calw(len(v))
    npr = numpy.array(range(len(v)))
    return scipy.signal.lombscargle(npr, v, w,normalize=norm)

def calw(n):
    w=[]
    for j in range(1,int(n/2)):
        w.append(2*pi*j/n)
    return numpy.array(w)

def ana(x,y):
    nor = True #False
    defr = pern(x,nor) - pern(y,nor)
    totpen = sum(defr**2)
    #return sqrt(totpen)
    return totpen 

def mass(amount,trms,lengh,burn,permode,name,savedatas):
    print("saving ",savedatas)
    number = str(random.randint(0,1000))
    setname = ["sta","wave","3ar"]
    if setname.count(name) == 0:
        data = pd.read_csv(name + ".csv")
    results = pd.DataFrame([])
    if permode == -1:
        percent = 0.1
    else:
        percent = permode 
    for i in range(amount):
        print(percent)
        print("epoch:",i,"/",amount)
        if setname.count(name) == 1:
            if trms.shape[0] == trms.shape[1]:
                data = create2d(trms,lengh,burn)
            else:
                data = create(trms,lengh,burn)
        print("1 of 4")
        withgap = mkhle(data.copy(),percent)
        print("2 of 4")
        refill = fill(withgap.copy())
        if savedatas:
            withgap.to_csv(name +  "-" + number + "-gap.csv")
            refill.to_csv(name +  "-" + number + "-fill.csv")
            data.to_csv(name +  "-" + number + "-data.csv")
        print("3 of 4")
        novar = data.shape[1]
        try:
            if novar == 1:
                res = fullana(refill,data,withgap)
            else:
                preres = []
                for vrs in range(novar):
                    preres.append(fullana2(refill.iloc[:,vrs],data.iloc[:,vrs],withgap.iloc[:,vrs]))
                res = pd.concat(preres)
                res.index = range(novar)
            res["trial"] = i
        except Exception as e:
            print(e,"#",novar)
            return withgap,refill,data
        print("4 of 4")
        if permode == -1:
            percent = percent + 0.1
        try:
            results = pd.concat([res,results])
        except Exception:
            print("reset")
            results = res
    print("exp number: " + number)
    results.to_csv(name +  "-" + str(int(permode*100)) + "-" + number + ".csv")
    return results

def fullana2(filleds,org,withgap):
    emas = ewmsug(withgap)
    ewms = emas.create_ewm()
    interps = withgap.copy().interpolate()
    spline = its.na_interpolation(withgap.copy(),option = "spline")
    stine = its.na_interpolation(withgap.copy(),option = "stine")
    sea = its.na_seadec(withgap.copy())
    kal = its.na_kalman(withgap.copy())
    seap = its.na_seasplit(withgap.copy())
    datalist = [
        spline,
        stine,
        sea,
        kal,
        seap,
        ewms.to_numpy(),
        interps.to_numpy(),
        filleds.to_numpy()
    ]
    table = []
    for i in range(len(datalist)):
        print(i,end="-")
        datalist[i] = maskorg(withgap,datalist[i])
        table.append(ana(org,datalist[i]))
    print("8")
    res = pd.DataFrame(table).transpose()
    res.columns = [
        "spline",
        "stine",
        "sea",
        "kal",
        "seap",
        "emas",
        "interps",
        "dualhi"
        ]
    return res

def plotana(data, name):
    trail = data.groupby(by=["trial"]).mean()
    error = data.groupby(by=["trial"]).std()
    for i in range(3):
        mp.errorbar(range(10,90,10),yerr=error.iloc[:,i],y=trail.iloc[:,i],label=trail.columns[i],linestyle="none",marker="o")
    mp.xlabel("% data missing")
    mp.ylabel("Mean square error of model vs orignal data")
    mp.legend(loc=2)
    mp.savefig(name + ".pdf")
    mp.show()

def get_trms(setname):
    if setname == "sta":
        trm1 =[0.6,0.4,-0.1,-0.05,0.03,0.008,0.0007,-0.0004,0.00002,]
        #trm2 =[0.57,0.41,-0.11,-0.05,0.03,0.009,0.0004,-0.0003,0.00001,]
        #trm3 =[0.63,0.37,-0.13,-0.04,0.032,0.0091,0.0008,-0.0005,0.00003,]
        trms = numpy.array([trm1])#,trm2,trm3])
    elif setname == "wave":
        trms = numpy.array(
            [[0.6,0.22,0.13,0.02,0.05,0.003,0.0004],
            [0.6,0.12,0.19,0.03,0.03,0.004,0.00041],
            [0.5,0.15,0.12,0.07,0.04,0.007,0.00042],
            [0.6,0.13,0.19,0.04,0.03,0.003,0.00043],
            [0.4,0.122,0.15,0.07,0.02,0.001,0.00044],
            [0.55,0.162,0.17,0.13,0.03,0.0045,0.00045],
            [0.45,0.152,0.12,0.07,0.01,0.0082,0.00046]
        ])
    elif setname == "3ar":
        trms = numpy.array([[0.9,-0.3,0.1]])
    else:
        trms = None
    return trms

def exp1(setname,per,lengh,savedatas):
    tms =  get_trms(setname)
    mass(lengh,tms,180,30,per,setname,savedatas)

if __name__ == "__main__":
    whats = input("filename ")
    try:
        hows = float(input("exptype "))
        whens = int(input("lenght "))
        wheres = bool(input("save data (0/1) "))
    except Exception:
        print("wrong input")
    else:
        exp1(setname=whats,per=hows,lengh=whens,savedatas=wheres)