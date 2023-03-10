import cvxpy as cp
import numpy as np


def hankel_imputaion_1d(data, lag, e, mask, predata=None,**kawgs):
    """
    Preforms a hankel imputaion with 1 dimetions
    by a convex optimation of minimation of the norm of hankle matrix of imputed variables
    with constraints = norm of mask of data and imputed variables less than a value e
    """
    print("monovar filling")
    N = len(data)
    Yapp = cp.Variable(N)
    try:
        Yapp.value = predata.transpose()[0]
    except Exception:
        print("no pre-data")
    bacYapp = Yapp[::-1]
    foward = cphanker(Yapp[:lag], Yapp[lag:])
    backard = cphanker(bacYapp[:lag], bacYapp[lag:])
    objective = cp.Minimize(cp.normNuc(foward) + cp.normNuc(backard))
    constraints = [cp.norm((data[mask] - Yapp[mask])) <= e]
    prob = cp.Problem(objective, constraints)
    prob.solve(**kawgs)
    return Yapp.value

def cphanker(row, reman):
    """
    creates a hankel matix with one dimetions
    """
    hank = []
    xi = row.shape[0]
    for i in range(xi):
        hank.append(np.append(row[: xi - i], reman[1 : xi + i + 1]))
    return cp.bmat(hank)


def construct1d(data, lag):
    """
    sends a ValueError if lag is too big
    """
    N = data.shape
    Yapp = cp.Variable(N)
    cp.Minimize(cp.normNuc(cphanker(Yapp[:lag], Yapp[lag:])))
