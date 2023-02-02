import cvxpy as cp
import numpy as np

def hankel_imputaion_1d(data,lag,e,mask,maxinter):
    """

    """
    N=len(data)
    Yapp = cp.Variable(N)
    objective = cp.Minimize(cp.normNuc(cphanker(Yapp[:lag],Yapp[lag:N])))
    constraints = [cp.norm((data[mask]-Yapp[mask])) <= e]
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True,max_iters=maxinter)
    return Yapp.value

def cphanker(row,reman):
    hank = []
    xi = row.shape[0]
    for i in range(xi):
        hank.append(np.append(row[:xi-i],reman[1:xi+i+1]))
    return cp.bmat(hank)

def construct1d(data,lag):
    """
    sends a ValueError if lag is too big
    """
    N= data.shape
    Yapp = cp.Variable(N)
    cp.Minimize(cp.normNuc(cphanker(Yapp[:lag],Yapp[lag:])))
