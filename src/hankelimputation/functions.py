import cvxpy as cp
import numpy as np


def hankel_imputaion(data, lag, e, mask, dim=1, double=True, predata=None,**kawgs):
    """
    by a convex optimation of minimation of the norm of hankle matrix of imputed variables
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
            Yapp.value = predata
    except Exception:
        print("no pre-data")
    foward = cphanker(Yapp[:lag], Yapp[lag:],dim)
    if double:
        bacYapp = Yapp[::-1]
        backard = cphanker(bacYapp[:lag], bacYapp[lag:],dim)
        objective = cp.Minimize(cp.normNuc(foward) + cp.normNuc(backard))
    else:
        objective = cp.Minimize(cp.normNuc(foward))
    constraints = [cp.norm((data[mask] - Yapp[mask])) <= e]
    prob = cp.Problem(objective, constraints)
    prob.solve(**kawgs)
    return Yapp.value

def cphanker(row, reman, dim):
    """
    creates a hankel matix with one dimetions
    """
    hank = []
    xi = row.shape[0]
    for i in range(xi):
        if dim <= 1:
            hank.append(np.append(row[: xi - i], reman[1 : xi + i + 1]))
        else:
            fiscal = xi - i
            lacal = xi + i + 1 
            left = row[:fiscal]#.reshape(-1)
            right = reman[dim:lacal]#.reshape(-1)
            mat_row = *left,*right
            hank.append(mat_row)
    return cp.bmat(hank)

def construct(data, lag, dim):
    """
    sends a ValueError if lag is too big
    """
    N = data.shape
    Yapp = cp.Variable(N)
    cp.normNuc(cphanker(Yapp[:lag], Yapp[lag:], dim))
