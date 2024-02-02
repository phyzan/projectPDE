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
    A BoundaryConditions object is independent of a grid. The matrix and array are constructed in methods, where
    the grid is a parameter.
    '''
    def __init__(self, pointsets: list[locus.Locus] = [], funcs: list[Callable[..., float]] = [], coefs: list[tuple[float]] = []):
        '''
        pointsets: list of Locus objects (Line, Surface, Point)
        funcs: list of functions, each corresponding to a pointset. The number of each function's parameters
            should be equal to the degrees of freedom of the pointset, unless the function is time-dependent:
            e.g for a Line, f(u) and f(u, t) is acceptable, but f(u, v) or just f(t) is not.
        coefs: list of 3 numbers [a, b, c]. The general type of boundary conditions is:
            a*f(...) + b*df_up/dn(...) + c*df_down/dn(...), where the last only apply if the dof of a pointset are
            1 less than the grid dimensions.
        '''
        if any([type(l) is locus.Locus for l in pointsets]):
            raise ValueError('Pointset items must be instanciated through a Locus subclass each (e.g Point, Line)')
        if len(funcs) != len(pointsets) or len(coefs) != len(pointsets):
            raise ValueError('The number of pointsets, funcs and coefficients provided must be the same')
        funcs = list(funcs)
        for i in range(len(funcs)):
            if tools.isnumeric(funcs[i]):
                funcs[i] = tools._floatfunc(funcs[i])
                
        self.ps = pointsets
        self.funcs = funcs
        self.coefs = coefs
        self.is_timedependent = [tools._is_timedependent(f) for f in funcs]
        self.has_timedependence = np.any(self.is_timedependent)
    
    def __add__(self, other: BoundaryConditions):
        return BoundaryConditions(self.ps+other.ps, self.funcs+other.funcs, self.coefs+other.coefs)

    def array(self, grid: grids.Grid, t: float = None):
        '''
        1D array with entries only in the indices that correspond to the given (possibly time-dependent) boundary conditions.
        For example, in 1D with boundary conditions f(0) = 2, f'(3) = 4 and if dx = 1: array = [2, 0, 0, 4]

        parameters
        --------------
        grid: grid on which the boundary conditions should be discretized
        t: If the boundary conditions are time dependent, pass the time parameter

        Returns
        -------------
        1D numpy array
        '''
        self._assert_grid_compatibility(grid)
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
        '''
        Every time-dependent or time-independent problem (e.g Wave equation or Poisson equation) is accompanied
        by some boundary conditions, in the grid boundaries, and possibly some in the interior of the grid.
        These boundary conditions are represented by a matrix.
        For example, consider the simple case of a 1D problem, in a small grid with 5 nodes and dx = 1
        Dirichlet conditions on both edges:
            m = [[1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1],]
        Neumann conditions on both edges:
            m = [[-1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, -1, 1],]
        Dirichlet left, Neumann right:
            m = [[1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, -1, 1],]
        
        In the Neumann case, forward finite differences are used on the left edge, and backward finite differences on the right edge.
        If acc > 1, then finite difference of higher order are used, and the matrix above will be different in the edges with Neumann conditions.
        This is generalized in the same manner for higher dimensions, or even boundary conditions in the interior of the grid

        





        Parameters
        -------------
        grid: The grid on which the boundary conditions should be discretized on
        acc: The accuracy that should be used in the finite differences for Neumann or Robin boundary conditions.

        Returns
        ---------------
        Sparce matrix with entries only on the rows that correspond to the nodes with boundary conditions
        '''
        self._assert_grid_compatibility(grid)
        m = grid.empty_matrix()
        for i in range(len(self.ps)):
            m += self.ps[i].robin_matrix(grid, self.coefs[i], acc)
        return m
    
    def reserved_nodes(self, grid: grids.Grid):
        '''
        When some boundary conditions are discretized on a grid, some nodes of this grid should not take place in
        the solution of a pde as regular nodes, but their values should be predetermined e.g for Dirichlet boundary conditions.
        Then, when solving a PDE, since a linear system of equations must be solved, the PDE must know which matrix entries
        and which array (rhs side) entries should stay empty, and be filled with the entries that correspond to the boundary conditions.
        For example, in the 1D case, for a grid with 10 nodes, nodes 0 and 9 are reserved. If there are more than one independent
        boundary conditionson a single point, then 2 nodes will be reserved, but this conserns higher order problems (e.g d^3f/dx^3 = 0)


        Parameters
        --------------
        grid: The grid on which the boundary conditions should be discretized on

        Returns
        --------------
        Iterable of grid nodes
        '''
        #some nodes might be repeated, but this is okay.
        #(e.g for the corners of a 2D grid, because the boundary conditions are given separately for each side)
        self._assert_grid_compatibility(grid)
        nodes = []
        for ps in self.ps:
            nodes.append(ps.nodes(grid))
        return itertools.chain(*nodes)
     
    def filter_matrix(self, grid: grids.Grid):
        '''
        Parameters
        --------------
        grid: The grid on which the boundary conditions should be discretized on

        Returns
        --------------
        Identity matrix (sparse format), whose entries that correspond to the reserved nodes have been removed.
        '''
        diag = np.ones(grid.n_tot, dtype=int)
        for node in self.reserved_nodes(grid):
            k = grid.flatten_index(node)
            diag[k] = 0
        return sp.dia_matrix((diag, 0), shape=(grid.n_tot, grid.n_tot))

    def _assert_grid_compatibility(self, grid: grids.Grid):
        for i in range(len(self.ps)):
            if self.ps[i].nd != grid.nd:
                raise ValueError('Grid dimensions not compatible with the dimensions of the shapes contained in the boundary conditions')

class _AbstractEdgeBC:
    '''
    This class is responsible only for storing variables while boundary conditions in
    the edges of a grid are being instanciated through the class AxisBcs.
    '''
    def __init__(self, func, coefs):

        self.func = func
        self.coefs = coefs


class AxisBcs:
    '''
    This class provides a more friendly way to instanciate boundary conditions
    in the boundaries of a rectangular space. Another way would be through
    the combination of classes locus.GridBoundary and BoundaryConditions,
    but that would be more difficult and itme consuming.
    '''
    def __init__(self, lower: _AbstractEdgeBC = None, upper: _AbstractEdgeBC = None, axis: int = 0):
        '''
        Parameters
        ---------------
        lower: The boundary condition at the lower end of a grid's axis.
            The parameter should be created through a Dirichlet(), Neumann() or Robin()
            command, whose parameters only include the value at the boundary
        upper: The boundary condition at the upper end of a grid's axis
        axis: The axis that these boundary condititions should be applied on
        '''
        self.bcs = [[lower, upper]]
        self.axis = [axis]
        self.nd = 1

    def __add__(self, other: AxisBcs):
        if type(other) is AxisBcs:
            self.bcs += other.bcs
            self.axis += other.axis
            self.nd += other.nd
        else:
            raise ValueError('Can only add AxisBcs to other AxisBcs. Use .set() when all boundary conditions in the grid edges have been specified')
        return self

    def set(self):
        '''
        When all boundary conditions in the grid boundaries have been given through 'AxisBcs',
        use this command to convert them into a BoundaryConditions object
        '''
        pss, funcs, coefs = [], [], []
        for i in range(self.nd):
            for j in range(2):
                if self.bcs[i][j] is not None:
                    pss.append(locus.GridBoundary(axis=self.axis[i], bound=j, nd=self.nd))
                    funcs.append(self.bcs[i][j].func)
                    coefs.append(self.bcs[i][j].coefs)
        return BoundaryConditions(pointsets=pss, funcs=funcs, coefs=coefs)


def ivp_operator(order: int, dx: float):
    '''
    Consider the initial value problem f(0) = a, f'(0) = b, f''(0) = c
    These conditions are sufficient to calculate the function f
    at the first 3 time steps, separated by equal lengths dt.
    Using 1st order forward finite differences, the linear system that should be solved is

    1/dt**0  | 1, 0, 0|   |  f(0) |      | f(0)  |
    1/dt**1  |-1, 1, 0| * | f(dt) | =    | f'(0) |
    1/dt**2  | 1,-2, 1|   |f(2*dt)|      | f''(0)|

    Then, f(0), f(dt), f(2*dt) can be calculated by applying the inverse matrix on the l.h.s
    on the initial conditions. This function calculates that inverse matrix


    Parameters
    --------------
    order: Order of the highest derivative in the initial conditions
    dx: Step of integration
    '''
    A = np.zeros((order, order))
    for i in range(order):
        A[i][:i+1] = findiff.fd_coefs(order=i, offsets=np.arange(0, i+1))/dx**i
    return np.linalg.inv(A)


def Robin(coefs: tuple[float], *args)->BoundaryConditions|_AbstractEdgeBC:
    '''
    Parameters
    -------------
    coefs (3-array): Coefficients of robin boundary conditions:
        e.g. coefs = [1, 2, 3] means 1*f(s) + 2*df/dn_{up} + 3*df/dn_{down}
    args: Should contain (not mandatory):
        Locus object (e.g Point, Line, Surface). If not given, the result must then be passed into the AxisBcs class
        function that will be evaluated on this object. Iif not given, it will be zero

    returns
    --------------
    BoundaryConditions object, or object that must be passed into the AxisBcs class
    to instanciate boundary conditions at the grid edges.


    Notes:
    
    The last 2 of the 3 robin coefficients only make sense when the boundary's
    degrees of freedom are one less that the grid dimensions.
    e.g For a 2D grid, the boundary should be a line
        For a 3D grid, the boundary should be a surface
    
    In that case, consider the direction perpendicular to the boundary
    (for a Line object, the direction is calculated using the right hand rule).
    Then,
        df/dn_up means df/dn right above the boundary
        df/dn_down means df/dn right below the boundary. This is the same as -df(d(-n))

    '''
    nargs = len(args)
    if nargs == 0:
        return _AbstractEdgeBC(tools._nullfunc, coefs)
    elif nargs == 1:
        arg = args[0]
        if callable(arg) or tools.isnumeric(arg):
            return _AbstractEdgeBC(arg, coefs)
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
    '''
    For "args" values, see Robin()
    
    Dirichlet boundary conditions consern the value of a function at a boundary.
    This means that we want 1*f(boundary) + 0*df/dn_{up} + 0*df/dn_{down}
    The Robin coefficints are [1, 0, 0]
    '''
    return Robin([1, 0, 0], *args)

def Neumann(*args):
    '''
    For "args" values, see Robin()
    
    Neumann boundary conditions consern the directional derivative on a boundary
    along the direction perpendicular to it. The perpendicular direction needs
    to be uniquely defined.
    '''
    return Robin([0, 1, 0], *args)

def derivative_discontinuity(*args):
    '''
    For "args" values, see Robin()
    
    This is a generalization of Neumann boundary conditions for boundary lines or Surfaces
    in the interior of a grid.
    Call when the quantity a*df/dn_up + b*df/dn_down is known on the boundary
    '''
    return Robin([0, 1, -1], *args)