import cvxpy as cp
import numpy as np
import cmath

def cphmat(vector,lag,N):
    """
    Make a hankel matrix from a vector

    """
    # caluate the number or rows
    try:
        no_of_rows = 1 + N[0] - lag
    except:
        no_of_rows = 1 + N - lag
    col = []
    for i in range(no_of_rows):
        row = []
        for j in range(lag):#crating the hankel matix one row at a time
            row.append(vector[i + j])
        col.append(row)
    return cp.bmat(col)#compining the rows in a matrix


def hankel_imputaion(data, lag, e, mask, dim=1, predata=None,lastrowsrange=[1,0],**kawgs):
    """
    by a convex optimation of minimisation of the norm of hankle matrix of imputed variables
    with constraints = norm of mask of data and imputed variables less than a value e
    """
    #caluate the size of the data
    if dim <= 1:
        print("monovar filling")
        N = len(data)
    else:
        print("Multivar filling")
        N = data.shape
    Yapp = cp.Variable(N)
    #attenpt to prefill data
    try:
        if dim <= 1:
            Yapp.value = predata.transpose()[0]
        else:
            Yapp.value = predata[:N[0]]
    except Exception as essa:
        print("no pre-data because of",essa,"\n")
    #create the hankel matix with the adjusbable varibles = adjutvar
    adjutvar = cphmat(Yapp,lag,N)
    # declare the objactive of min(nuclear norm(adjutvar))
    objective = cp.Minimize(cp.normNuc(adjutvar))
    if len(e) == 1: # with only the containt of norm(real_data - adjusbable_varibles)
        constraints = [cp.norm((data[mask] - Yapp[mask])) <= e[0]] 
    else:# with also the fouirer contrait
        constraints = [cp.norm((data[mask] - Yapp[mask])) <= e[0],contraint2(N,Yapp,e[1],lastrowsrange)]   
    prob = cp.Problem(objective, constraints)# define problem
    prob.solve(**kawgs)# send to cvx api to solve using scs
    return Yapp.value# return the value of the convex solution

def contraint2(N,originalvalues,epsilon2,lastrowsrange):
    """
    calualte the fourtier contraint
    """
    fullmat = cp.matmul(formatix(N,True),originalvalues)
    contrval = cp.norm(fullmat[N-lastrowsrange[0]:N-lastrowsrange[1]]) #select last few rows
    return (contrval <= epsilon2)

def formatix(n,pure=False):
    """
    calulate the full fourtier matix
    """
    mat = []
    for c in range(n):
        row = []
        for r in range(n):
            val = mkw(n)**(c*r)
            if pure:
                row.append(val)
            else:
                row.append(np.ma.round(val))
        mat.append(row)
    return cp.bmat(mat)

def mkw(bign):
    z = -1j
    trm = z*2*cmath.pi
    w = cmath.exp(trm/bign)
    return w


def construct(data, lag):
    """
    sends a ValueError if lag is too big
    """
    N = data.shape
    Yapp = cp.Variable(N)
    cp.normNuc(cphmat(Yapp, lag, N))
