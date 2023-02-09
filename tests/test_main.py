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
    trm =[
        0.6,
        0.4,
        -0.1,
        -0.05,
        0.03,
        0.008,
        0.0007,
        -0.0004,
        0.00002,
    ]
    y = [0]*len(trm)
    a = loopar(y.copy(),trm,40,0.1)
    b = loopar(y.copy(),trm,40,0.2)
    ar = pandas.DataFrame([a,b]).transpose()
    ar.to_csv("test.csv")
    holear = mkhle(ar.copy(),0.1)
    filled = hankelimputation.processing(holear,0,0.1,5000)
    assert filled.isna().sum().sum() == 0

def test_batch():
    trm =[
        0.6,
        0.4,
        -0.1,
        -0.05,
        0.03,
        0.008,
        0.0007,
        -0.0004,
        0.00002,
    ]
    y = [0]*len(trm)
    a = loopar(y.copy(),trm,30,0.1)
    b = loopar(y.copy(),trm,30,0.2)
    ar = pandas.DataFrame([a,b]).transpose()
    holear = mkhle(ar.copy(),0.1)
    filled = hankelimputation.processing(holear,10,0.1,5000)
    assert filled.isna().sum().sum() == 0

def test_single():
    trm =[
        0.6,
        0.4,
        -0.1,
        -0.05,
        0.03,
        0.008,
        0.0007,
        -0.0004,
        0.00002,
    ]
    y = [0]*len(trm)
    a = loopar(y.copy(),trm,40,0.1)
    ar = pandas.DataFrame(a)
    holear = mkhle(ar.copy(),0.1)
    filled = hankelimputation.processing(holear,0,0.1,5000)
    assert filled.isna().sum().sum() == 0