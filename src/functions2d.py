import cvxpy as cp

def mcwb2d(Y,L,e,bina,maxinter,lag):
    """
    Weighted vectors with forecast
    """
    N= Y.shape
    Yapp = cp.Variable(N)
    constraints = [cp.norm((Y[bina]-Yapp[bina])) <= e]
    objective = cp.Minimize(cp.normNuc(cp2dhanker(Yapp[:L],Yapp[L:],lag)))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True,max_iters=maxinter)
    return Yapp.value

def cp2dhanker(row,reman,lag):
    hank = []
    xi = row.shape[0]
    for i in range(xi):
        fiscal = xi - i
        lacal = xi + i + 1
        left = cp.hstack(row[:fiscal])
        right = cp.hstack(reman[lag:lacal])
        mat_row = cp.hstack([left,right])
        hank.append(mat_row)
    return cp.bmat(hank)

def construct2d(Y,L,lag):
    """
    sends a ValueError if L is too big
    """
    N= Y.shape
    Yapp = cp.Variable(N)
    cp.Minimize(cp.normNuc(cp2dhanker(Yapp[:L],Yapp[L:],lag)))
