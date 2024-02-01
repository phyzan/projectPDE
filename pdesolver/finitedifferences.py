import numpy as np
import scipy.sparse as sp
import math
from . import grids
from. import tools


def fd_offsets(n: int, node: int, order: int, acc: int = 1, periodic: bool = False):
    '''
    Returns the offset points that will be used to calculate
    the finite difference coefficients, given the order and accuracy
    of differentiation. Priority will be given to central differences,
    but the offset indices will deviate from that close to the grid boundaries
    unless the grid is periodic


    Arguments:
    -------------
    n: total number of points in the one dimensional grid
    i: the index of the grid that we want to get the offsets
    order: order of differentiation
    acc: accuracy of differentiation

    Returns
    --------------
    list of offsets: e.g np.array([-2, -1, 0, 1, 2])

    Then, i + offsets are the stencil points used in the finited differences
    '''
    p = nof_cfd_offsets(order, acc)
    if (node >= p and n-1-node >= p) or periodic is True:
        offsets = np.arange(-p, p+1, dtype=int) #equivalent to cfd_offsets(...)
    else:
        ns = order + acc
        if node < p:
            offsets = np.arange(0, ns) - node
        else:# n-1-i < p
            offsets = np.arange(n-ns, n) - node

    return offsets

def ffd_offsets(n: int, node: int, order: int, acc: int = 1, periodic: bool = False):
    ns = order+acc
    if (node+ns-1 > n-1 and periodic is False) or not (0 <= node <= n-1):
        raise ValueError('Not enough grid points for forward finite difference or node not in grid bounds')
    return np.arange(0, ns, dtype=int)

def bfd_offsets(n: int, node: int, order: int, acc: int = 1, periodic: bool = False):
    ns = order+acc
    if (node-ns+1 < 0 and periodic is False) or not (0 <= node <= n-1):
        raise ValueError('Not enough grid points for backward finite difference or node not in grid bounds')
    return np.arange(-ns+1, 1, dtype=int)

def fd_coefs(order: int, offsets: list):
    '''
    Creates the finite difference coefficients to approximate any n-th order derivative
    of a discretized function

    Arguments
    -------------
    order (int): order of differentiation
    oddsets (tuple of ints): Indices around a given point to be used in the
        differentiation approximantion

    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    '''
    n = len(offsets)
    offsets = np.array(offsets)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i] = offsets**i
    b = np.zeros(n)
    b[order] = math.factorial(order)
    return np.linalg.solve(A, b)

def nof_cfd_offsets(order: int, acc: int = 1):
    return int((order+1)/2)-1+acc

def cfd_coefs(order: int, acc: int = 1):
    '''
    Arguments
    ----------
    order (int): Order of differentiation
    acc (int): accuracy of finite difference

    Returns
    ----------
    cfd_coefs (array): central finite difference coefficients
    p (int): number of offset elements from center

    The length of cfd_coefs is 2*p + 1

    '''
    p = nof_cfd_offsets(order, acc)
    offsets = np.arange(-p, p+1, dtype=int)
    return fd_coefs(order, offsets)

def ffd_coefs(order: int, acc: int = 1):
    offsets = np.arange(0, order+acc, dtype=int)
    return fd_coefs(order, offsets)

def bfd_coefs(order: int, acc: int = 1):
    offsets = np.arange(-order+acc+1, 1, dtype=int)
    return fd_coefs(order, offsets)


def fd_nodes(n: int, node: int, order: int, acc: int = 1, periodic: bool = False):
    offsets = fd_offsets(n=n, node=node, order=order, acc=acc, periodic=periodic)
    return (offsets + node + n) % n

_fd_map = {0: fd_offsets, 1: ffd_offsets, -1: bfd_offsets}

def diff_element(n: int, node: int, dx: float, order: int, acc: int, periodic: bool, fd: int = 0):
    offsets = _fd_map[fd](n=n, node=node, order=order, acc=acc, periodic=periodic)
    nf = len(offsets)
    rows = node + np.zeros(nf, dtype=int)
    cols = (rows + offsets + n) % n #equavalent to: cols = fd_nodes(...)
    vals = fd_coefs(order=order, offsets=offsets)/dx**order
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

def identity_element(grid: grids.Grid, node: tuple[int]):
    row = [grid.flatten_index(node)]
    return sp.csr_matrix(([1], (row, row)), shape=(grid.n_tot, grid.n_tot))

def partial_diff_element(grid: grids.Grid, node: tuple[int], axis: int, order: int, acc: int, fd: int = 0):
    offsets_1d = _fd_map[fd](n=grid.shape[axis], node=node[axis], order=order, acc=acc, periodic=grid.periodic[axis])
    nods_1d = (node[axis] + offsets_1d + grid.shape[axis]) % grid.shape[axis]

    nf = len(offsets_1d)
    rows = grid.flatten_index(node) + np.zeros(nf, dtype=int)
    node = tuple(node)
    cols = []
    for i in range(nf):
        _node = node[:axis] + (nods_1d[i],) + node[axis+1:]
        cols.append(grid.flatten_index(_node))
    vals = fd_coefs(order=order, offsets=offsets_1d)/grid.dx[axis]**order
    return sp.csr_matrix((vals, (rows, cols)), shape=(grid.n_tot, grid.n_tot))

def directional_diff_element(grid: grids.Grid, node: tuple[int], direction: tuple[float], order: int=1, acc: int=1, fd: int = 0):
    vec = direction/np.linalg.norm(direction)
    m = grid.empty_matrix()
    kwargs = dict(grid=grid, node=node, order=order, acc=acc)
    for i in range(grid.nd):
        m += vec[i] * partial_diff_element(**kwargs, axis=i, fd=fd*tools.sign(vec[i]))
    return m

def directional_diff_along_line_element(grid: grids.Grid, node: tuple[int], direction: tuple[int], order: int=1, acc: int=1, fd: int = 0):
    '''directional derivative'''
    n, i = grids.maximum_nodes_along_dir(grid.shape, node, direction)
    offsets = _fd_map[fd](n, i, order, acc, periodic = all(grid.periodic))
    nf = len(offsets)
    rows = np.zeros(nf, dtype=int) + grid.flatten_index(node)

    direction = np.array(direction)
    cols = rows + np.array([grid.flatten_index(f*direction) for f in offsets])

    dr = np.linalg.norm(grid.dx*direction)
    vals = fd_coefs(order, offsets)/dr**order
    return sp.csr_matrix((vals, (rows, cols)), shape=(grid.n_tot, grid.n_tot))