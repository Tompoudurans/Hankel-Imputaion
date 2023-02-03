import cvxpy as cp


def hankel_imputaion_2d(data, lag, e, mask, maxinter, dim):
    """
    Preforms a hankel imputaion with 2 dimetions
    by a convex optimation of minimation of the norm of hankle matrix of imputed variables
    with constraints = norm of mask of data and imputed variables less than a value e
    """
    N = data.shape
    Yapp = cp.Variable(N)
    constraints = [cp.norm((data[mask] - Yapp[mask])) <= e]
    objective = cp.Minimize(cp.normNuc(cp2dhanker(Yapp[:lag], Yapp[lag:], dim)))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, max_iters=maxinter)
    return Yapp.value


def cp2dhanker(row, reman, dim):
    """
    creates a hankel matix with two dimetions
    """
    hank = []
    xi = row.shape[0]
    for i in range(xi):
        fiscal = xi - i
        lacal = xi + i + 1
        left = cp.hstack(row[:fiscal])
        right = cp.hstack(reman[dim:lacal])
        mat_row = cp.hstack([left, right])
        hank.append(mat_row)
    return cp.bmat(hank)


def construct2d(data, lag, dim):
    """
    sends a ValueError if lag is too big
    """
    N = data.shape
    Yapp = cp.Variable(N)
    cp.Minimize(cp.normNuc(cp2dhanker(Yapp[:lag], Yapp[lag:], dim)))
