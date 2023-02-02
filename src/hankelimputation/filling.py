from .functions1d import *
from .functions2d import *
import pandas as pd

def testmax_Lag(data):
    """
    Test the Maximum number that lag can have a large lag gives a better results but too big have this error occurs:
    ValueError: All the input dimensions except for axis 0 must match exactly. cp.bmat(hank)
    """
    numpy_data = data.to_numpy()
    N,dim = data.shape
    pnt = int(N/2.8)
    for v in range(pnt):
        lag = pnt - v
        try:
            print(lag)
            if dim > 1:
                construct2d(numpy_data,lag,dim)
            else:
                construct1d(numpy_data,lag)
        except ValueError:
            if lag % 5 == 0:
                print("less than",lag)
        else:
            print(lag)
            return lag

def batchfilling(data,filename,batch_size):
    """
    fill data in batches using hankel imputaion method and saves it to a csv file
    """
    numpy_data = data.to_numpy()
    mask = data.notna()
    lag = testmax_Lag(data.iloc[:batch_size])
    N,dim = data.shape
    list_filled_dataframe = []
    for batch in range(0,len(data)-batch_size,batch_size):
        print(batch,batch+batch_size)
        if dim > 1:
            filled = hankel_imputaion_2d(numpy_data[batch:batch+batch_size],lag,0.1,mask[batch:batch+batch_size],50000,dim)
        else:
            filled = hankel_imputaion_1d(numpy_data[batch:batch+batch_size],lag,0.1,mask[batch:batch+batch_size],50000)
        filled_dataframe = pd.DataFrame(filled)
        list_filled_dataframe.append(filled_dataframe)
    togat = pd.concat(list_filled_dataframe)
    togat.to_csv(filename + "_filled.csv",index=False)
    return list_filled_dataframe

def fullfilling(data,filename):
    """
    fill data using hankel imputaion method and saves it to a csv file
    """
    lag = testmax_Lag(data)
    N,dim = data.shape
    numpy_data = data.to_numpy()
    mask = data.notna()
    if dim > 1:
        filled = hankel_imputaion_2d(numpy_data,lag,0.1,mask,50000,dim)
    else:
        filled = hankel_imputaion_1d(numpy_data.transpose()[0],lag,0.1,mask.to_numpy().transpose()[0],50000)
    filled_dataframe = pd.DataFrame(filled)
    filled_dataframe.to_csv(filename + "_filled.csv",index=False)
    return filled