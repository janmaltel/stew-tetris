import numpy as np
from numba import njit
import scipy.optimize as optim


@njit
def stew_reg(X, y, D, lam):
    return np.linalg.inv(X.T @ X + lam * D) @ X.T @ y


def stew_loss(beta, X, y, D, lam):
    residuals = y - X @ beta
    l = residuals.T.dot(residuals) + lam * beta.T.dot(D).dot(beta)
    return l


def stew_grad(beta, X, y, D, lam):
    return 2 * np.dot(beta, X.T).dot(X) - 2 * y.T.dot(X) + 2 * lam * beta.dot(D)


def stew_hessian(beta, X, y, D, lam):
    return 2 * X.T.dot(X) + 2 * lam * D


def stew_reg_iter(X, y, D, lam, method='Newton-CG'):
    op = optim.minimize(fun=stew_loss, x0=np.zeros(X.shape[1]), args=(X, y, D, lam),
                        jac=stew_grad, hess=stew_hessian, method=method)
    return op.x

