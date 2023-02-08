from .functions1d import *
from .functions2d import *
import pandas as pd


def testmax_Lag(numpy_data):
    """
    Test the Maximum number that lag can have a large lag gives a better results but too big have this error occurs:
    ValueError: All the input dimensions except for axis 0 must match exactly. cp.bmat(hank)
    """
    N, dim = numpy_data.shape
    pnt = int(N / 2.8)
    for v in range(pnt):
        lag = pnt - v
        try:
            if dim > 1:
                construct2d(numpy_data, lag, dim)
            else:
                construct1d(numpy_data.transpose()[0], lag)
        except ValueError as e:
            if lag % 5 == 0:
                print("less than", lag)
            if lag < int(N / 10):
                print(e)
                return
        else:
            print(lag)
            return lag


def batchfilling(numpy_data, mask, lag, e, dim, batch_size, N):
    """
    fill data in batches using hankel imputaion method
    """
    list_filled_dataframe = []
    for batch in range(0, N - batch_size, batch_size):
        print(batch, batch + batch_size)
        filled_dataframe = filling(
            numpy_data[batch : batch + batch_size],
            mask[batch : batch + batch_size],
            lag,
            e,
            dim,
        )
        list_filled_dataframe.append(filled_dataframe)
    return pd.concat(list_filled_dataframe)


def filling(numpy_data, mask, lag, e, dim):
    """
    fill data using hankel imputaion method
    """
    if dim > 1:
        filled = hankel_imputaion_2d(numpy_data, lag, e, mask, 50000, dim)
    else:
        filled = hankel_imputaion_1d(
            numpy_data.transpose()[0], lag, e, mask.to_numpy().transpose()[0], 50000
        )
    return pd.DataFrame(filled)


def processing(data, batch, e):
    """
    calulates the mask, the lag, the shape
    """
    numpy_data = data.to_numpy()
    mask = data.notna()
    N, dim = data.shape
    if batch == 0:
        lag = testmax_Lag(numpy_data)
        filled = filling(numpy_data, mask, lag, e, dim)
    else:
        lag = testmax_Lag(numpy_data[:batch])
        filled = batchfilling(numpy_data, mask, lag, e, dim, batch, N)
    return filled


def refillzero(dataframe):
    refill = dataframe.where(dataframe != 0.0)
    print("failed amount:", refill.isna().sum())
    return refill.interpolate()
