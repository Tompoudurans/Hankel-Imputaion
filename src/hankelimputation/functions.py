import cvxpy as cp
import numpy as np

def cphmat(vec,hoz,dim,N):
    """
    hmat Hankel matrix from a vector "vec"

    """
    try:
        vert = 1 + N[0] - hoz
    except:
        vert = 1 + N - hoz
    col = []
    for i in range(vert):
        row = []
        for j in range(hoz):
            row.append(vec[i + j])
        col.append(row)
    return cp.bmat(col)


def hankel_imputaion(data, lag, e, mask, dim=1, double=True, predata=None,**kawgs):
    """
    by a convex optimation of minimisation of the norm of hankle matrix of imputed variables
    with constraints = norm of mask of data and imputed variables less than a value e
    """
    if dim <= 1:
        print("monovar filling")
        N = len(data)
    else:
        print("Multivar filling")
        N = data.shape
    Yapp = cp.Variable(N)
    try:
        if dim <= 1:
            Yapp.value = predata.transpose()[0]
        else:
            Yapp.value = predata[:N[0]]
    except Exception as essa:
        print("no pre-data because of",essa,"\n")
    foward = cphmat(Yapp,lag,dim,N)
    objective = cp.Minimize(cp.normNuc(foward))
    constraints = [cp.norm((data[mask] - Yapp[mask])) <= e]   
    prob = cp.Problem(objective, constraints)
    prob.solve(**kawgs)
    return Yapp.value

def construct(data, lag, dim):
    """
    sends a ValueError if lag is too big
    """
    N = data.shape
    Yapp = cp.Variable(N)
    #cp.normNuc(cphanker(Yapp[:lag], Yapp[lag:], dim)
    cp.normNuc(cphmat(Yapp, lag, dim, N))
