from __future__ import annotations
from . import grids
from . import finitedifferences as findiff
from . import tools
import numpy as np
from typing import Callable
from typing import Iterable
import scipy.sparse as sp
import scipy.misc as spm
import matplotlib.pyplot as plt


class Locus:

    def __init__(self, nd, dof):
        self.nd = nd
        self.dof = dof

    def param_values(self, grid: grids.Grid)->Iterable:
        pass

    def nodes(self, grid: grids.Grid):
        pass

    def x(self, *u)->tuple:
        pass

    def normal_vector(self, *u)->tuple:
        pass

    def identity_matrix(self, grid: grids.Grid):
        m = grid.empty_matrix()
        for node in self.nodes(grid):
            m += findiff.identity_element(grid, node)
        return m

    def normaldiff_matrix(self, grid: grids.Grid, order: int, fd: int, acc: int)->sp.csr_matrix:
        '''
        GridBBoundary is the only class that implements its own function because it is faster
        '''
        m = grid.empty_matrix()
        for u in self.param_values(grid):
            nod = grid.node(*self.x(*u))
            vec = self.normal_vector(*u)
            m += findiff.directional_diff_element(grid=grid, node=nod, direction=vec, order=order,acc=acc, fd=fd)
        return m

    def robin_matrix(self, grid: grids.Grid, coefs: list[float], acc: int):
        '''
        Given their normal direction, if we name f_up the field above the boundary
        and f_d the field below it (where "up" means along the normal direction)
        we have the "robin" operator: a*f + b*df_up/dn + c*df_down/dn, useful in
        electrostatics for example, where dV_up/dn - dV_down/dn ~ charge density
        a, b, c are the coefs in the function arguments
        '''
        m = grid.empty_matrix()
        for i in range(3):
            if coefs[i] != 0:
                if i == 0:# no need for this if statement, but it's faster
                    m += coefs[i] * self.identity_matrix(grid)
                else:
                    fd = 1 if i == 1 else -1
                    m += coefs[i] * self.normaldiff_matrix(grid, order=1, fd=fd, acc=acc)
        return m
    
    def plot(self, grid: grids.Grid):
        self._assert_compatibility(grid)
        fig, ax = plt.subplots()
        ax.set_xlim(*grid.limits[0])
        if self.nd == 1:
            for node in self.nodes(grid):
                ax.scatter(*grid.coords(node), 0, s=3, c='k')
        if self.nd == 2:
            ax.set_ylim(*grid.limits[1])
            for node in self.nodes(grid):
                ax.scatter(*grid.coords(node), s=3, c='k')
        else:
            pass # TODO

        return fig, ax

    def _assert_compatibility(self, grid: grids.Grid):
        if self.nd != grid.nd:
            raise ValueError('Locus and grid lie on different dimensions')
        

class Point(Locus):

    dof = 0

    def __init__(self, *coords):
        if not all([tools.isnumeric(n) for n in coords]):
            raise ValueError('Point arguments must be numerical coordinates')
        self.coords = coords
        super().__init__(nd=len(coords), dof=0)

    def __repr__(self):
        return f'Point{self.coords}'
    
    def nodes(self, grid: grids.Grid):
        self._assert_compatibility(grid)
        return grid.node(*self.coords),
    
    def param_values(self, grid: grids.Grid):
        return [()]

    def x(self):
        return self.coords
    
    def normal_vector(self):
        if self.nd != 1:
            raise ValueError('A point must lie on a 1d line so that is has a uniquely defined normal vector')
        return (1,)


class Line(Locus):

    dof = 1

    def __init__(self, x: list[Callable[[float], float]], lims):
        self._x = x
        self.lims = lims
        super().__init__(nd=len(x), dof=1)

    def __repr__(self):
        return f'Line({self.nd}d-space, u = {self.lims})'

    def x(self, u):
        return [xi(u) for xi in self._x]
    
    def xdot(self, u):
        return [spm.derivative(xi, u, 1e-6) for xi in self._x]
    
    def param_values(self, grid: grids.Grid)->Iterable:
        self._assert_compatibility(grid)
        x = self.x
        a, b = self.lims
        nod = grid.node(*x(a))
        nods = [nod]
        u_arr = [[a]]
        neighbors = grid.neighboring_nodes(nod)
        while grid.node(*x(self.lims[1])) != nod or len(u_arr) == 1:
            u = (a+b)/2
            nod = grid.node(*x(u))
            if nod in neighbors:
                if nod in nods:
                    break
                else:
                    neighbors = grid.neighboring_nodes(nod)
                    b = self.lims[1]
                    u_arr.append([u])
                    nods.append(nod)
            elif nod == nods[-1]:
                a = u
            else:
                b = u
        return u_arr
    
    def nodes(self, grid: grids.Grid):
        nods = []
        for u in self.param_values(grid):
            nods.append(grid.node(*self.x(*u)))
        return nods

    def normal_vector(self, u):
        if self.nd != 2:
            raise ValueError('A line must lie on a 2d plane so that is has a uniquely defined normal vector')
        tangent_vec = self.xdot(u)
        return normal_vector(tangent_vec)
        
    def reverse(self):
        return Line(*self._x, lims=(self.lims[1], self.lims[0]))


class Surface(Locus):
    #TODO
    dof = 2

    pass


class GridBoundary(Locus):

    _bound = {0: 'lower', 1: 'upper'}

    def __init__(self, axis: int, bound: int[0, 1], nd: int):
        self.axis = axis
        self.bound = bound
        super().__init__(nd=nd, dof=nd-1)

    def __repr__(self):
        return f'Boundary(axis = {self.axis}, {self._bound[self.bound]})'

    def param_values(self, grid: grids.Grid)->Iterable:
        self._assert_compatibility(grid)
        '''
        The curve parameters of a grid boundary are the coordinates of the grid at its boundary
        '''
        for ijk in self.nodes(grid):
            x = grid.coords(ijk)
            yield *x[:self.axis], *x[self.axis+1:]

    def x(self, *params):
        raise ValueError("Grid boundaries do not have predetermined coordinate values, they depends on each grid's limits")

    def nodes(self, grid: grids.Grid):
        self._assert_compatibility(grid)
        return grid.edge(self.axis, bound=self.bound, corners=True)
    
    def normaldiff_matrix(self, grid: grids.Grid, order: int, fd: int = 1, acc: int = 1)->sp.csr_matrix:
        if fd == -1:
            raise ValueError('Grid edges must automate betweem central, forward and backward finite differences')
        self._assert_compatibility(grid)
        n, dx, per = grid.shape[self.axis], grid.dx[self.axis], grid.periodic[self.axis]
        node = self.bound*(n-1)
        m = findiff.diff_element(n=n, node=node, dx=dx, order=order, acc=acc, periodic=per)
        return tools.operate_on(axis=self.axis, shape=grid.shape, matrix=m, edges=True)



def normal_vector(*vecs):
    m = np.array(vecs).transpose()
    nd = len(vecs) + 1 #==len(vecs[i]) for all i
    if m.shape != (nd, nd-1):
        raise ValueError('You need n-1 vectors to generate a vector perpendicular to them in an n-dimensional space')
    
    v = np.zeros(nd)
    m = np.array(vecs).transpose()
    for i in range(nd):
        v[i] = (-1)**i*np.linalg.det(np.vstack((m[:i], m[i+1:])))
    return v


def Circle(r: float, center: tuple[float])->Line:
    
    def x(u):
        return center[0] + r*np.cos(u)
    
    def y(u):
        return center[1] + r*np.sin(u)
    
    return Line([x, y], lims=(0, 2*np.pi))


def Square(a: float, start: tuple[float]):

    x0, y0 = start
    def x(u):
        if 0 <= u < 1:
            return x0 + u*a
        elif 1 <= u < 2:
            return x0 + a
        elif 2 <= u < 3:
            return x0 + a*(3-u)
        else:
            return x0
        
    def y(u):
        if 0 <= u < 1:
            return y0
        elif 1 <= u < 2:
            return y0 + (u-1)*a
        elif 2 <= u < 3:
            return y0 + a
        else:
            return y0 + a*(4-u)
        
    return Line([x, y], lims=(0, 4))