from .functions import *
import pandas as pd


def testmax_Lag(numpy_data,N,dim):
    """
    Test the Maximum number that lag can have a large lag gives a better results but too big have this error occurs:
    ValueError: All the input dimensions except for axis 0 must match exactly. cp.bmat(hank)
    """
    pnt = int(N / 2.8)
    for v in range(pnt):
        lag = pnt - v
        try:
            if dim > 1:
                construct(numpy_data, lag, dim)
            elif dim == 1:
                construct(numpy_data.transpose()[0], lag, dim)
        except ValueError as e:
            if lag % 5 == 0:
                print("less than", lag)
            if lag < int(N / 10):
                print(e)
                return
        else:
            print(lag)
            return lag


def batchfilling(numpy_data, mask, e, dim, batch_size, N, **kwags):
    """
    fill data in batches using hankel imputaion method
    """
    list_filled_dataframe = []
    lag = testmax_Lag(numpy_data[:batch_size],batch_size,dim)
    for batch in range(0, N, batch_size):
        endbatch = min(batch + batch_size,N)
        newbat = endbatch - batch
        if newbat > 1:
            print(batch, endbatch)
            if newbat < batch_size:
            	lag = testmax_Lag(numpy_data[:newbat],newbat,dim)
            filled_dataframe = filling(
                numpy_data[batch : endbatch],
                mask[batch : endbatch],
                lag,
                e,
                dim,
                **kwags,
            )
            list_filled_dataframe.append(filled_dataframe)
    return pd.concat(list_filled_dataframe)


def filling(numpy_data, mask, lag, e, dim, **kawgs):
    """
    fill data using hankel imputaion method
    """
    if dim > 1:
        filled = hankel_imputaion(numpy_data, lag, e, mask, dim, **kawgs)
    elif dim == 1:
        filled = hankel_imputaion(
            numpy_data.transpose()[0], lag, e, mask.to_numpy().transpose()[0], **kawgs
        )
    else:
        filled = hankel_imputaion(
            numpy_data, lag, e, mask, **kawgs
        )
    return pd.DataFrame(filled)


def processing(data, batch, e, **kawgs):
    """
    calulates the mask, the lag, the shape
    """
    print("epsilon:",e)
    numpy_data = data.to_numpy()
    mask = data.notna()
    predata = data.copy().interpolate().to_numpy()
    try:
        N, dim = data.shape
    except ValueError:
         N = len(data)
         dim = 0
    print("your data has",N,"timesteps and",dim,"varibles")
    if batch == 0:
        lag = testmax_Lag(numpy_data,N,dim)
        filled = filling(numpy_data, mask, lag, e, dim, predata=predata, **kawgs)
    else:
        filled = batchfilling(numpy_data, mask, e, dim, batch, N, predata=predata, **kawgs)
    return filled


def refillzero(dataframe):
    refill = dataframe.where(dataframe != 0.0)
    print("failed amount:", refill.isna().sum())
    return refill.interpolate()
