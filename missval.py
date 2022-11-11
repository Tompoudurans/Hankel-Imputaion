from math import pi,cos,sqrt
import numpy as np
import scipy.linalg
import cvxpy as cp


#### ISSUES:
# orignal program does not work with missing values dispite what title says 
# what does ./ mean?

def froweightold(L,K):
    N=L+K-1
    w = []
    for i in range(1,L-1):
        w.append(i)
    for i in range(L,K):
        w.append(L)
    for i in range(K+1,N):
        w.append(N-i+1)
    return w
    

def hmat(vec,vert):
    """
    hmat Hankel matrix from a vector "vec"

    """
    hoz = 1 + len(vec) - vert
    col = []
    for i in range(vert):
        row = np.array([])
        for j in range(hoz):
            row = np.append(row, vec[i + j,])
        col.append(row)
    return np.array(col)

def create_data(N):
    s = []
    for n in range(N):
        s.append(cos(2*pi*n/10)) # case 1
        #s(n)=cos(2*pi*n/10)*exp(0.02*n) # case 2
    return s


def hankvec_avg(X):
    """
    hankvec Perform diagonal averaging 
    """
    L,K = X.shape
    Xbase = np.append(X,np.zeros(K*K))
    bmat = Xbase[K:].reshape((K+L-1), K)
    y = []
    fws = froweights(K,L)
    print(bmat.shape,len(fws))
    for i in range(bmat.shape[0]):
        y.append(sum(bmat[i,:])/fws[i])
    return np.array(y)

def lra(X,r):
    [U,S,V2]= scipy.linalg.svd(X,full_matrices=False) # U,S,V] = SVD(X,"econ") produces the "economy size"
    S2 = S*np.identity(len(S))
    for i in range(r,len(S)):
        S2[i,i] = 0 #add 0s and the end for forcasting - replace by a mask instead?
    print(U.shape,S2.shape,S.shape,V2.shape)
    return np.matmul(U,np.matmul(S2,V2))#

def mcwf(Y,L,M,w,e):
    """
    Weighted vectors with forecast
    """
    N=len(Y);
    #cvx_begin sdp;
    #variable Yapp(N+M);
    x = N+M
    Yapp = cp.Variable(x)
    objective = cp.Minimize(cp.normNuc(scipy.linalg.hankel(Yapp[:L],Yapp[L:N+M])[0][0]))
    constraints = np.linalg.norm(sqrt(w)*(Y[:N]-Yapp[:N])) <= e
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result
    
def mcw(Y,L,w,e):
    """
    Weighted vectors
    """
    N=len(Y);
    #cvx_begin sdp;
    #%cvx_solver mosek;
    #%cvx_precision low;
    Yapp = cp.variable(N);
    objective = cp.minimize(cp.normNuc(scipy.linalg.hankel(Yapp[:L],Yapp[L:N])))
    constraints = cp.norm(sqrt[w]*(Y-Yapp)) <=e 
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result

def hyperpend(vec,a,pos):
    """
    append like matlab. should look a way to avoid using this
    """
    try:
        vec[pos] = a
    except IndexError:
        vec.append(a)
    return vec

def wvnorm(y,w):
    """
    wvnorm Compute weighted vector norm given a vector y and a 
    vector of weights w
    """
    tot = []
    for i in range(len(w)):
        tot.append(w[i]*(y[i]**2))
    return sum(tot)

def froweights(L,K):
    N=L+K-1
    w = []
    for i in range(1,L-1):
        w = hyperpend(w,i,i)
    for i in range(L,K):
        w = hyperpend(w,L,i)
    for i in range(K,N):
        w = hyperpend(w,N-i+1,i)
    return np.array(w)

def hyperpend(vec,a,pos):
    try:
        vec[pos] = a
    except IndexError:
        vec.append(a)
    return vec

def main(
    Y,
    L=10, # L =rows and k =colums n is size m is missing 
    k=2,
    m = 15,
    ):
    Ya = []
    last = len(Y)-m
    X=hmat(Y[:last],L)
    wF=froweights(L,X.shape[1])
    end = len(Y)
    Ftotrans = Y[:end-m]
    Ftohankav = lra(X,k)
    Ftonorm = hankvec_avg(Ftohankav)-Ftotrans.transpose()
    Ftosqrt = wvnorm(Ftonorm,wF)
    tauF=sqrt(Ftosqrt)
    Ya=mcwf(Ftotrans.transpose(),L,m,wF,tauF)
    return Ya

if False:
    N = 50
    s = create_data(N)
    main(s,N)