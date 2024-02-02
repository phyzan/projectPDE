from __future__ import annotations
from . import grids
from . import fields
from . import finitedifferences as fd
from . import tools
import numpy as np
import scipy.sparse as sp
from typing import Callable
import inspect


def diff_matrix(order: int, n: int, dx: float, acc: int=1, periodic: bool = False):
    m = sp.csr_matrix((n, n))
    for i in range(n):
        m += fd.diff_element(n=n, node=i, dx=dx, order=order, acc=acc, periodic=periodic, fd=0)
    return m

def partialdiff_matrix(grid: grids.Grid, order: int, axis: int, acc: int = 1):
    '''
    Function that creates sparse matrices representing discretized partial derivative operators

    Arguments
    -----------
    order (int): The order of the derivative operator
    shape (tuple): The shape of the corresponding grid that a discretized function lies at
    axis (int): The axis (variable) of the partial derivative
    dx (float): The grid spacing along the given axis
    acc (int): The accuracy of the differentiation (see fd_coefs and cfd_coefs)
    edges (bool): Whether or not to apply the procedure at the two edges of the axis

    Returns
    ---------------
    Sparse differentiation matrix
    '''
    diff = diff_matrix(order=order, n=grid.shape[axis], dx=grid.dx[axis], acc=acc, periodic=grid.periodic[axis])
    return tools.operate_on(axis=axis, shape=grid.shape, matrix=diff, edges=True)

def Gradient(nd: int):
    diffs = [Diff(order=1, axis=i) for i in range(nd)]
    return VectorOperator(components=tuple(diffs))

def asoperator(a)->Operator:
    if isinstance(a, Operator):
        return a
    elif tools.isnumeric(a):

        def matrix(grid: grids.Grid, acc: int):
            return a*sp.identity(grid.n_tot)
        
        return Operator(matrix=matrix, signature=str(a))
    elif callable(a):
        return Function(a)
    else:
        raise ValueError(f'Object of type {type(a)} cannot be represented as an operator')


class Operator:
    '''
    Base class for all operators. Should not be called by the user usually, this might change.
    '''
    def __init__(self, matrix: Callable[[grids.Grid, int], sp.csr_matrix], signature: str):
        '''
        Arguments
        ---------

        grid (grids.Grid): The grid that the operator should lie at
        generator (callable): A function of 'acc' (int) and 'edges' (bool) that returns a sparse matrix,
            corresponding to a discretized operator
        signature (str): A string representing the operator (e.g. 'd/dx' for a differential operator)
        '''
        self.signature = signature
        self._matrix = matrix

    def __add__(self, other: Operator|float):
        other = asoperator(other)

        def add(grid: grids.Grid, acc: int):
            return self.matrix(grid=grid, acc=acc) + other.matrix(grid=grid, acc=acc)

        return Operator(add, self._coupled_signature(other, '+'))

    def __sub__(self, other: Operator|float):
        other = asoperator(other)

        def sub(grid: grids.Grid, acc: int):
            return self.matrix(grid=grid, acc=acc) - other.matrix(grid=grid, acc=acc)

        return Operator(sub, self._coupled_signature(other, '-'))

    def __mul__(self, other: Operator|float|VectorOperator)->Operator:
        if type(other) is  VectorOperator:
            return VectorOperator.__rmul__(other, self)
        other = asoperator(other)

        def mul(grid: grids.Grid, acc: int):
            return self.matrix(grid=grid, acc=acc).dot(other.matrix(grid=grid, acc=acc))

        return Operator(mul, self._coupled_signature(other, '*'))
    
    def __neg__(self):
        op = Operator.__mul__(self, -1)
        op.signature = f'-{self.signature}'
        return op
    
    def __radd__(self, other: Operator|float):
        return Operator.__add__(self, other)

    def __rsub__(self, other: Operator|float):
        op = - Operator.__sub__(self, other)
        op.signature = f'{asoperator(other).signature} - {self.signature}'
        return op

    def __rmul__(self, other: Operator|float):
        other = asoperator(other)
        return other * self

    def __pow__(self, power: int):
        a = self
        for _ in range(1, power):
            a *= self
        a.signature = f'{self.signature}**{power}'
        return a
    
    def apply(self, field: fields.ScalarField, acc: int = 1):
        '''
        Apply the operator on a field object (discretized function)

        Arguments
        --------------
        arr (ndarray): Discretized function
        acc (int): Accuracy of the discretized operator (applies only for differential operators)

        returns
        --------------
        ndarray of the resulting function
        '''
        new_field = self.matrix(field.grid, acc=acc).dot(field.field.flatten(order='F'))

        return fields.ScalarField(grid=field.grid, field=new_field.reshape(field.grid.shape, order='F'))

    def matrix(self, grid: grids.Grid, acc: int = 1)->sp.csr_matrix:
        '''
        Arguments
        ------------
        acc (int): Accuracy of the discretized operator (applies only for differential operators)
        '''
        return self._matrix(grid=grid, acc=acc)

    def __str__(self):
        '''
        Will print the default matrix: acc=1
        '''
        return self.signature
    
    def _coupled_signature(self, op: Operator|float, bind: str):
        if isinstance(op, Operator):
            return f' {bind} '.join([self.signature, op.signature])
        elif tools.isnumeric(op):
            return f' {bind} '.join([self.signature, str(op)])
        else:
            raise ValueError('"op" argument can only be Operator or numeric')
        

class VectorOperator:
    def __init__(self, components: tuple[Operator]):
        if not all([isinstance(op, Operator) for op in components]):
            raise ValueError('VectorOperator components must be operators')
        self.components = components
        self.nd = len(components)

    def __getitem__(self, i)->Operator:
        return self.components[i]

    def __mul__(self, other: Operator|VectorOperator):
        if type(other) is Operator:
            return VectorOperator([self[i]*other for i in range(self.nd)])
        elif type(other) is VectorOperator:
            return sum([self[i]*other[i] for i in range(self.nd)])
        else:
            return self * asoperator(other)
        
    def __rmul__(self, other: Operator):
        if type(other) is Operator:
            return VectorOperator([other * self[i] for i in range(self.nd)])
        else:
            _other = asoperator(other)
            return _other * self

    def apply(self, field: fields.ScalarField, acc: int = 1):
        components = [self[i].apply(field=field, acc=acc) for i in range(self.nd)]
        return fields.VectorField(components=components)

    def cross(self, field: fields.VectorField, acc: int = 1):
        if self.nd != 3 or field.nd != 3:
            raise ValueError('Cross product only implemented in 3 dimensions')

        components = []
        for i in range(3):
            j, k = (i+1)%3, (i+2)%3
            components.append(self[j].apply(field[k], acc=acc) - self[k].apply(field[j], acc=acc))
        return fields.VectorField(components=components)

    def dot(self, field: fields.VectorField, acc: int = 1):
        if self.nd != field.nd:
            raise ValueError('Operator cannot be applied on a grid of these dimensions')
        
        s = 0
        for i in range(self.nd):
            s += self[i].apply(field=field[i], acc=acc).field
        
        return fields.ScalarField(grid=field.grid, field=s)

class Diff(Operator):
    '''
    Discretized differential operator
    '''
    def __init__(self, order: int, axis: int = 0):
        '''
        Arguments
        ----------
        grid (grids.Grid): The grid that the operator should lie at
        order (int): derivative order
        axis (int): The axis (variable) of the partial derivative
        '''
        if order == 1:
            signature = ['d/dx', 'd/dy', 'd/dz']
        else:
            signature = [f'd^{order}/dx^{order}', f'd^{order}/dy^{order}', f'd^{order}/dz^{order}']
        self.order = order
        self.axis = axis

        def diff(grid: grids.Grid, acc: int = 1):
            return partialdiff_matrix(grid=grid, order=order, axis=axis, acc=acc)
        
        super().__init__(matrix=diff, signature=signature[axis])


class Laplacian(Operator):
    def __init__(self):
        
        def matrix(grid: grids.Grid, acc: int):
            m = grid.empty_matrix()
            for i in range(grid.nd):
               m += Diff(order=2, axis=i).matrix(grid=grid, acc=acc)
            return m

        super().__init__(matrix=matrix, signature='Δ')


class Function(Operator):
    '''
    Function operator. When applied to an array of a discretized function,
    returns the product of the two functions
    '''
    def __init__(self, f: Callable[..., float|np.ndarray]):
        '''
        grid (grids.Grid): The grid that the operator should lie at.
        f (callable): The function of the grid coordinates (e.g f(x, y) for a 2d grid)
        '''
        self.nargs = len(inspect.signature(f).parameters)

        def matrix(grid: grids.Grid, acc: int = 1):
            if self.nargs != grid.nd:
                raise ValueError('Function arguments and grid shape are inconsistent')
            diag = f(*grid.x_mesh()).flatten(order='F')
            n = grid.n_tot
            return sp.dia_matrix((diag, 0), (n,n)).tocsr()
        
        super().__init__(matrix=matrix, signature=f.__name__)

        self.f = f
    
    def __call__(self, *args):
        return self.f(*args)
    
class IdentityOperator(Operator):

    _instance = None

    def __new__(cls):
        if cls._instance is not None:
            return cls._instance
        else:
            cls._instance = super().__new__(cls)
            return cls._instance
        
    def __init__(self):

        def matrix(grid: grids.Grid, acc: int = 1):
            return sp.identity(grid.n_tot)

        super().__init__(matrix=matrix, signature='1')

I = asoperator(1)