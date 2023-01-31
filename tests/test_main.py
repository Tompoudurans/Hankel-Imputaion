import hankelimputation
import random
import numpy
import pandas

def mkhle(data,percent):
    x, y = data.shape
    count = 0
    aim = round(x * y * percent)
    while count < aim:
        row = random.randint(0, x - 1)
        data.iloc[row, :] = numpy.nan
        count = data.isna().sum().sum()
        print(count, "(", aim, ")/", x * y, sep="")
    return data

def loopar(y,trm,non,sigma):
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
            return y
    return y

def test_full():
    trm = [0.3,-0,1,0.9]
    y = [0]*len(trm)
    ar = pandas.DataFrame([loopar(y,trm,200,0.1),loopar(y,trm,200,0.2)],"test.csv")
    holear = mkhle(ar,0.1)
    hankelimputation.fullfilling(holear,"test")
    
