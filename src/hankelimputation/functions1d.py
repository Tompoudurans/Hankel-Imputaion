import cvxpy as cp
import numpy as np

def hankel_imputaion_1d(data,lag,e,mask,maxinter):
    """
    Preforms a hankel imputaion with 1 dimetions
    by a convex optimation of minimation of the norm of hankle matrix of imputed variables
    with constraints = norm of mask of data and imputed variables less than a value e
    """
    N=len(data)
    Yapp = cp.Variable(N)
    objective = cp.Minimize(cp.normNuc(cphanker(Yapp[:lag],Yapp[lag:N])))
    constraints = [cp.norm((data[mask]-Yapp[mask])) <= e]
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True,max_iters=maxinter)
    return Yapp.value

def cphanker(row,reman):
    """
    creates a hankel matix with one dimetions
    """
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
