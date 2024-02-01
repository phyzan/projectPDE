from __future__ import annotations
import itertools
from typing import Literal
import numpy as np
import scipy.sparse as sp
from . import tools


def flatten_index(index: tuple[int], shape: tuple[int], order: Literal['F', 'C']):
    '''
    Arguments
    ------------
    index (tuple): index of multidimensional matrix
    shape (tuple): dimensions of matrix

    Returns
    ------------
    index of corresponding flattened array
    '''
    if len(index) != len(shape):
        raise ValueError("'Shape' and 'index' shape mismatch")
    nd = len(shape)
    res = 0
    if order == 'C':
        for k in range(nd):
            res += index[k]*tools.prod(shape[k+1:])
    else:
        for k in range(nd):
            res += index[k]*tools.prod(shape[:k])        
    return res

def reshape_toindex(k: int, shape: tuple, order: Literal['F', 'C']):
    '''
    Reverse procedure of unravel_index

    Arguments
    -------------
    k: Unraveled index of a flattened array
    shape (n_1, n_2, n_3): Shape of corresponding grid

    Returns
    -------------
    multidimensional index of corresponding grid

    '''
    nd = len(shape)
    ijk = nd*[0]
    _k = k
    if order == 'C':
        for i in range(nd):
            ijk[i] = _k // tools.prod(shape[i+1:])
            _k -= ijk[i]*tools.prod(shape[i+1:])
    elif order == 'F':
        for i in range(nd):
            ijk[nd-1-i] = _k // tools.prod(shape[:nd-1-i])
            _k -= ijk[nd-1-i]*tools.prod(shape[:nd-1-i])        
    else:
        raise ValueError('Acceptable values for "order" are "F" or "C"')
    return ijk

def maximum_nodes_along_dir(shape: tuple[int], point: tuple[int], direction: tuple[int]):
    '''
    Given an N-dimensional grid, we can extend a line along a direction (base vector with integers) from a specific point
    only until the line meets the grid boundaries.
    e.g
    The "0" is located at point = (8, 2). The line along the direction (1, 2) from that point reaches a total of 4 points
    (3 "o", and 1 more including "0"). Then, the "0" is located at index 1 along this direction. If we choose the opposite direction, (-1, -2),
    the "0" is located at index = 2
    . . . . . . . . . . . .
    . . . . . . . . . . o .
    . . . . . . . . . . . .
    . . . . . . . . . o . .
    . . . . . . . . . . . .
    . . . . . . . . 0 . . .
    . . . . . . . . . . . .
    . . . . . . . o . . . .

    Parameters:
    -------------
    shape: shape of grid (n_x, n_y, n_z)
    point: coordinates of a point (i_x, i_y, i_z)
    direction: base vector for direction e.g (2, 0, 1)

    Returns
    ------------
    n: number of nodes along the given direction (but both ways), including the given point
    i: index of point, if we put these nodes side by side
    '''
    nd = len(shape)
    _point, _direction, _shape = [], [], []
    for i in range(nd):
        if direction[i] != 0:
            _point.append(point[i])
            _direction.append(direction[i])
            _shape.append(shape[i])
    _nd = len(_shape)
    
    l_pos, l_neg = [], []
    for i in range(_nd):
        sgn = [np.sign(_direction[i]), np.sign(-_point[i]*_direction[i])]
        l1 = sgn[0]*((_shape[i]-_point[i]-1)//abs(_direction[i]))
        l2 = sgn[1]*(abs(-_point[i])//abs(_direction[i]))
        l_neg.append(min(l1, l2))
        l_pos.append(max(l1, l2))
    
    n, i = min(l_pos) - max(l_neg) + 1, -max(l_neg)
    return n, i

class Iterator:
    '''
    Creates objects that iterates through all combinations of integers
    from given ranges. It differs from itertools.product on the fact
    that it starts the iteration with the left item icreasing fastest,
    and since it only conserns integers, it does not need entire arrays of them,
    but only their start and end points
    '''
    __slots__ = ['limits', 'range', 'nd', 'n', 'n_tot', 'item', '_progress']

    def __init__(self, order: Literal['F', 'C'], *ranges):
        self.limits = ranges
        self.nd = len(ranges)
        self.n = tuple([ranges[i][1]-ranges[i][0]+1 for i in range(self.nd)])
        self.n_tot = tools.prod(self.n)
        self.item = [ranges[i][1] for i in range(self.nd)]
        self._progress = self.n_tot-1
        if order == 'F':
            self.range = range(self.nd)
        elif order == 'C':
            self.range = range(self.nd-1, -1, -1)

    def next(self):
        if self._progress < self.n_tot-1:
            for i in self.range:
                if self.item[i] == self.limits[i][1]:
                    self.item[i] = self.limits[i][0]
                else:
                    self.item[i] += 1
                    self._progress += 1
                    break
        else:
            self.item = [self.limits[i][0] for i in range(self.nd)]
            self._progress = 0
        return tuple(self.item)

    def __iter__(self):
        for _ in range(self.n_tot):
            yield self.next()


class Grid:
    '''
    Class for creating an N-dimensional cartesian grid,
    to be used for discretizing the domain of partial differential equations.
    '''

    __slots__ = ['shape', 'nd', 'limits', 'periodic', 'identity', 'x', 'dx', 'n_tot', 'n_bounds', 'n_goodbounds', 'n_corners']

    def __init__(self, shape: tuple, limits: tuple[tuple], periodic: tuple[bool] = None):
        '''
        Arguments
        -----------
        shape (tuple): The number of points in its axis in reverse order (e.g (nz, ny, nx))
        limits (tuple): The limits of each axis (in usual order: (xlims, ylims)). e.g. ((0, 10), (0, 20))
        '''
        nd = len(shape)
        if nd != len(limits):
            raise ValueError('Shape and limits incompatible')
        for l in limits:
            if len(l) != 2:
                raise ValueError('Axis limits format not supported')
        if periodic is None:
            periodic = tuple(nd*[False])
        elif isinstance(periodic, tuple):
            if len(periodic) != nd or not all([type(p) is bool for p in periodic]):
                raise ValueError('"Periodic" argument format not supported')
        else:
            raise ValueError('"Periodic" argument should be None or tuple[bool]')
        
        self.identity = dict(shape=shape, limits=limits, periodic=periodic) #all we need to distinguish between grids
        #the next three attributes are the identity of the grid
        self.shape = shape
        self.limits = limits
        self.periodic = periodic

        self.nd = nd
        self.x = [np.linspace(*limits[i], self.shape[i]) for i in range(nd)]
        self.dx = [(self.limits[i][1]-self.limits[i][0])/(self.shape[i]-1) for i in range(nd)]

        self.n_tot = tools.prod(shape)
        self.n_bounds = self.n_tot - tools.prod([i-2 for i in shape])
        self.n_goodbounds = 2*sum([tools.prod([j-2 for j in shape[:i]])*tools.prod([j-2 for j in shape[i+1:]]) for i in range(nd)])
        self.n_corners = self.n_bounds - self.n_goodbounds

    def __eq__(self, other: Grid):
        return self.identity == other.identity

    def coords(self, index: tuple[int]):
        '''
        Arguments
        ------------
        index (tuple): The index of the grid in the format (i, j, k)
        '''
        return tuple(self.x[axis][index[axis]] for axis in range(self.nd))

    def x_mesh(self):
        '''
        Create meshgrid of grid variables to passed into a funtion.
        e.g

        In 3D, if f = (x, y, z), then f(*self.x_mesh()).flatten() can be used
        in a Function operator, or for plotting (when reshaped)

        '''
        return np.meshgrid(*self.x, indexing='ij')
    
    def flatten_index(self, index: tuple[int]):
        return flatten_index(index, self.shape, order='F')
    
    def tonode(self, i: int):
        '''
        Argument: Index of flattened grid
        Returns: Reshaped index
        '''
        return reshape_toindex(i, self.shape, order='F')
    
    def nodes(self, edges: bool = True):
        '''
        Generator object that iterates through all (i, j, k) indices of the grid

        Arguments
        -------------
        include_bounds (bool): Whether or not to include the exterior indices in the iteration

        Notes
        -------
        Total number will be self.n_tot
        '''
        r = 1-edges
        ranges = [(r, self.shape[i]-1-r) for i in range(self.nd)]
        return Iterator('F', *ranges)

    def edge(self, axis: int, bound: int, corners: bool):
        if corners is True:
            r = [(0, s-1) for s in self.shape]
        else:
            r = [(1, s-2) for s in self.shape]
        return Iterator('F', *r[:axis], 2*[bound*(self.shape[axis]-1)], *r[axis+1:])

    def edges(self, corners: bool):
        '''
        Generator object to iterates through all boundary (i, j, k) indices of the grid


        Notes
        -------
        if corners is True:
            Total number is self.n_bounds
        if corners is False:
            Total number is self.n_goodbounds
        
        '''
        generators = []
        _all = [(0, s-1) for s in self.shape]
        _interior = [(1, s-2) for s in self.shape]
        _bound = [[2*[k*(s-1)] for k in range(2)] for s in self.shape]
        for d in range(self.nd-1, -1, -1):
            for k in range(2):
                if corners is True:
                    generators.append(Iterator('F', *_interior[:d], _bound[d][k], *_all[d+1:]))
                else:
                    generators.append(Iterator('F', *_interior[:d], _bound[d][k], *_interior[d+1:]))
        return itertools.chain(*generators)

    def corners(self):
        '''
        Notes
        ----------
        Total number is self.n_corners
        '''
        if self.nd == 1:
            return itertools.product([])
        else:
            _all = [(0, s-1) for s in self.shape]
            _interior = [(1, s-2) for s in self.shape]
            _bound = [[2*[k*(s-1)] for k in range(2)] for s in self.shape]
            generators = []
            for i in range(self.nd-1, 0, -1):
                for j in range(i-1, -1, -1):
                    for k1 in range(2):
                        for k2 in range(2):
                            generators.append(Iterator('F', *_interior[:j],_bound[j][k1], *_interior[j+1:i],_bound[i][k2],*_all[i+1:]))
        return itertools.chain(*generators)

    def node(self, *coords: float):
        '''
        coords = (x, y, z)
        returns nearest node
        '''
        for i in range(self.nd):
            if coords[i] < self.limits[i][0] or coords[i] > self.limits[i][1]:
                raise ValueError('Coordinates are not within the grid limits')

        nod = self.nd*[0]
        for i in range(self.nd):
            _n = (coords[i] - self.limits[i][0])
            _d = (self.limits[i][1] - self.limits[i][0])
            nod[i] = round(_n/_d*(self.shape[i] - 1)) #np.abs(self.x[i]-coords[i]).argmin()

        return tuple(nod)

    def neighboring_nodes(self, node: tuple):
        k = []
        for i in range(self.nd):
            if self.periodic[i] or 0 < node[i] < self.shape[i]-1:
                low = (self.shape[i]+node[i]-1) % self.shape[i]
                high = (node[i]+1) % self.shape[i]
                k.append([low, node[i], high])
            elif node[i] == 0:
                k.append([0, 1])
            else:
                k.append([self.shape[i]-2, self.shape[i]-1])
        nods = []
        for i in itertools.product(*k):
            if not all([i[j] == node[j] for j in range(self.nd)]):
                nods.append(i)
        return nods
    
    def empty_matrix(self):
        return sp.csr_matrix((self.n_tot, self.n_tot))

    def copy(self):
        '''
        Creates a copy of the grid. This is useful for classes that inherit from Grid.
        e.g. ScalarField(...).grid()
        '''
        return Grid(**self.identity)