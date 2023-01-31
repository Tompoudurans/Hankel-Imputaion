import cvxpy as cp
import numpy as np

def mcwb(Y,L,e,bina,maxinter):
    """
    Weighted vectors with forecast
    """
    N=len(Y)
    Yapp = cp.Variable(N)
    objective = cp.Minimize(cp.normNuc(cphanker(Yapp[:L],Yapp[L:N])))
    constraints = [cp.norm((Y[bina]-Yapp[bina])) <= e]
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True,max_iters=maxinter)
    return Yapp.value

def cphanker(row,reman):
    hank = []
    xi = row.shape[0]
    for i in range(xi):
        hank.append(np.append(row[:xi-i],reman[1:xi+i+1]))
    return cp.bmat(hank)

def construct1d(Y,L,lag):
    """
    sends a ValueError if L is too big
    """
    N= Y.shape
    Yapp = cp.Variable(N)
    cp.Minimize(cp.normNuc(cphanker(Yapp[:L],Yapp[L:],lag)))