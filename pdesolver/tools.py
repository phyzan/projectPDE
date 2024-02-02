from typing import Callable
import numpy as np
import scipy.sparse as sp
import os, sys


def prod(args):
    '''
    Argument
    ------------
    args (iterable): array of numbers

    Returns
    ------------
    Product of numbers inside 'args'
    '''
    a = 1
    for i in args:
        a *= i  
    return a

def _is_timedependent(func: Callable[..., float]):
    nargs = func.__code__.co_argcount
    args = func.__code__.co_varnames[:nargs]
    return 't' in args

def _nullfunc(*args):
    return 0

def _floatfunc(n):
    
    def f(*args):
        return n
    
    return f

def isnumeric(n):
    return np.issubdtype(type(n), np.number)

def sign(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0
    
def _output_progress(i, steps):
    if int(steps/1000) == 0:
        _print = True
    elif i % int(steps/1000) == 0:
        _print = True
    else:
        _print = False
    
    if _print:
        sys.stdout.flush()
        sys.stdout.write('\rComputing: {} %'.format(round(100*i/steps, 2), end=""))

def multi_kronecker_product(*matrices):
    """
    Calculate the multi Kronecker product of a list of sparse matrices.

    Parameters:
    - matrices (list): List of sparse matrices.

    Returns:
    - result (sparse matrix): Multi Kronecker product of the input matrices.
    """
    result = matrices[0]

    for matrix in matrices[1:]:
        result = sp.kron(result, matrix)

    return result

def operate_on(axis: int, shape: tuple, matrix: sp.spmatrix, edges: bool = True):
    '''
    Generalizes the action of a matrix operator to more dimensions, so that it can act
    on a flattened array that when reshaped, represents an N-dimensional grid.


    Arguments
    -------------
    axis (int): The axis of the multidimensional grid that the operator should act on
        axis = 0: 'x' axis
        axis = 1: 'y' axis
        axis = 2: 'z' axis
    shape (nx, ny, nz): The shape of the multidimensional grid (reverse order)
    matrix (sparse matrix): The matrix to be generalized
    edges (bool): Whether or not to apply the procedure at the two edges of the axis


    returns
    -------------
    Generalized matrix operator
    '''
    nd = len(shape)
    if shape[axis] != matrix.shape[0]:
        raise ValueError('Matrix shape mismatch')
    matrices = []
    for i in range(nd-1, -1, -1):
        if i == axis:
            matrices.append(matrix)
        else:
            n = shape[i]
            diag = np.ones(n)
            if not edges:
                diag[0], diag[n-1] = 0, 0
            matrices.append(sp.dia_matrix((diag, 0), shape=(n,n)))

    return multi_kronecker_product(*matrices)