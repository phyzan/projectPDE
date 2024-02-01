from __future__ import annotations
from . import grids
from . import locus
from . import finitedifferences as findiff
from . import tools
from typing import Callable
import numpy as np
import scipy.sparse as sp
import itertools


class BoundaryConditions:
    '''
    Class responsible for creating boundary conditions. BC objects build their own matrices and arrays
    with entries only in the rows that correspond to a set of points (e.g a line or a surface discretized
    on a given grid).
    '''
    def __init__(self, pointsets: tuple[locus.Locus], funcs: list[Callable[..., float]], coefs: tuple[tuple[float]]):
        '''
        pointsets: list of Locus objects (Line, Surface, Point)
        funcs: list of functions, each corresponding to a pointset. The number of each function's parameters
            should be equal to the degrees of freedom of the pointset, unless the function is time-dependent:
            e.g for a Line, f(u) and f(u, t) is acceptable, but f(u, v) or just f(t) is not.
        coefs: list of 3 numbers [a, b, c]. The general type of boundary conditions is:
            a*f(...) + b*df_up/dn(...) + c*df_down/dn(...), where the last only apply if the dof of a pointset are
            1 less than the grid dimensions.
        '''
        if any([type(l) is locus for l in pointsets]):
            raise ValueError('pointset items must be instanciated through a Locus subclass each')
        _funcs = list(funcs)
        for i in range(len(pointsets)):
            if not callable(funcs[i]):
                _funcs[i] = tools._floatfunc(funcs[i])
        self.ps = pointsets
        self.funcs = _funcs
        self.coefs = coefs
        self.is_timedependent = [tools._is_timedependent(f) for f in _funcs]
        self.has_timedependence = np.any(self.is_timedependent)
    
    def __add__(self, other: BoundaryConditions):
        #self.ps and other.ps are tuples of Locus subclasses. The tuples are added below
        return BoundaryConditions(self.ps+other.ps, self.funcs+other.funcs, self.coefs+other.coefs)

    def array(self, grid: grids.Grid, t:float = None):
        arr = np.zeros(grid.n_tot)
        for ps, f in zip(self.ps, self.funcs):
            for node, param_value in zip(ps.nodes(grid), ps.param_values(grid)):
                nod = grid.flatten_index(node)
                if tools._is_timedependent(f):
                    arr[nod] += f(*param_value, t)
                else:
                    arr[nod] += f(*param_value)
        return arr
    
    def matrix(self, grid: grids.Grid, acc: int = 1):
        m = grid.empty_matrix()
        for i in range(len(self.ps)):
            m += self.ps[i].robin_matrix(grid, self.coefs[i], acc)
        return m
    
    def reserved_nodes(self, grid: grids.Grid):
        #some nodes might be repeated, but this is okay.
        nodes = []
        for ps in self.ps:
            nodes.append(ps.nodes(grid))
        return itertools.chain(*nodes)
     
    def filter_matrix(self, grid: grids.Grid):
        diag = np.ones(grid.n_tot, dtype=int)
        for ijk in self.reserved_nodes(grid):
            k = grid.flatten_index(ijk)
            diag[k] = 0
        return sp.dia_matrix((diag, 0), shape=(grid.n_tot, grid.n_tot))


class _AbstractEdgeBc:
    def __init__(self, func: Callable[..., float], coefs: tuple[float]):
        self.func = func
        self.coefs = coefs


class Null:
    
    _instance = None

    def __new__(cls):
        if cls._instance is not None:
            return cls._instance
        else:
            null = super().__new__(cls)
            cls._instance = null
            return null
        
    def __init__(self):
        pass

def ics_operator(order, dt):
    A = np.zeros((order, order))
    for i in range(order):
        A[i][:i+1] = findiff.fd_coefs(order=i, offsets=np.arange(0, i+1))/dt**i
    return np.linalg.inv(A)

def Robin(coefs: tuple[float], *args):
    nargs = len(args)
    if nargs == 0:
        return _AbstractEdgeBc(func = tools._nullfunc, coefs = coefs)
    elif nargs == 1:
        arg = args[0]
        if callable(arg) or tools.isnumeric(arg):
            return _AbstractEdgeBc(func = arg, coefs = coefs)
        elif isinstance(arg, locus.Locus):
            return BoundaryConditions(pointsets = [arg], funcs = [tools._nullfunc], coefs = [coefs])
        else:
            raise ValueError('Unsupported arguments')
    elif nargs == 2:
        if callable(args[0]) or tools.isnumeric(args[0]):
            assert isinstance(args[1], locus.Locus)
            return BoundaryConditions(pointsets = [args[1]], funcs = [args[0]], coefs = [coefs])
        elif callable(args[1]) or tools.isnumeric(args[1]):
            assert isinstance(args[0], locus.Locus)
            return BoundaryConditions(pointsets = [args[0]], funcs = [args[1]], coefs = [coefs])
        else:
            raise ValueError('Unsupported arguments')
    else:
        raise ValueError('Unsupported arguments')

def Dirichlet(*args):
    return Robin([1, 0, 0], *args)

def Neumann(*args):
    return Robin([0, 1, 0], *args)

def derivative_discontinuity(*args):
    return Robin([0, 1, -1], *args)


def boundaryconditions(*bcs: tuple[_AbstractEdgeBc | Null] | _AbstractEdgeBc | Null)->BoundaryConditions:
    if len(bcs) == 2:
        if isinstance(bcs[0], _AbstractEdgeBc) or bcs[0] is Null():
            bc, nd = bcs
            return boundaryconditions(*[(bc, bc) for _ in range(nd)])
            
    pss, funcs, coefs = [], [], []
    for i in range(len(bcs)):
        for j in range(2):
            if isinstance(bcs[i][j], _AbstractEdgeBc):
                pss.append(locus.SingleGridBoundary(axis=i, bound=j, nd=len(bcs)))
                funcs.append(bcs[i][j].func)
                coefs.append(bcs[i][j].coefs)
            else:
                if bcs[i][j] != Null():
                    raise ValueError('Not supported argument in initializing grid boundaries')
                
    return BoundaryConditions(pointsets=pss, funcs=funcs, coefs=coefs)

