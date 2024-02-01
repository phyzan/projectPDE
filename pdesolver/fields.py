from __future__ import annotations
from .import grids
from . import operators
from . import finitedifferences as findiff
import scipy.interpolate as interp
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class ScalarField:
    def __init__(self, grid: grids.Grid, field: np.ndarray):
        if field.shape != grid.shape:
            raise ValueError('grids.Grid shape and array shape incompatible')
        
        self.grid = grid
        self.field = field
        self.nd = grid.nd
        self.x = grid.x

    def __call__(self, *coords):
        return interp.interpn(self.grid.x, self.field, list(coords), method='cubic')[0]
    
    def __add__(self, other: ScalarField):
        self._assert_compatibility(other)
        return ScalarField(self.grid, self.field+other.field)

    def __sub__(self, other: ScalarField):
        self._assert_compatibility(other)
        return ScalarField(self.grid, self.field-other.field)        

    def __mul__(self, other: ScalarField):
        self._assert_compatibility(other)
        return ScalarField(self.grid, self.field*other.field)
    
    def __eq__(self, other: ScalarField):
        return self.field == other.field
    
    def _assert_compatibility(self, other: ScalarField):
        if self.grid != other.grid:
            raise ValueError('Fields lie on different grids')
        
    def evaluate(self, axis: int, value: float)->ScalarField|float:
        _n = (value - self.grid.limits[axis][0])
        _d = (self.grid.limits[axis][1] - self.grid.limits[axis][0])
        index = round(_n/_d*(self.grid.shape[axis] - 1))
        slices = (slice(None),) * axis + (index,)+ (slice(None),) * (self.nd-1-axis)
        if self.nd == 1:
            return self.field[slices]
        else:
            new_identity = self.grid.identity.copy()
            for i in new_identity:
                new_identity[i] = new_identity[i][:axis] + new_identity[i][axis+1:]
            return ScalarField(grids.Grid(**new_identity), self.field[slices])
        
    def plot(self, **kwargs):
        fig, ax = plt.subplots()
        x = self.grid.x_mesh()
        if self.nd == 1:
            ax.plot(*x, self.field, **kwargs)
        elif self.nd == 2:
            ax.pcolormesh(*x, self.field, **kwargs)
            norm = plt.Normalize(self.field.min(), self.field.max())
            sm = cm.ScalarMappable(norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax)
        else:
            raise ValueError('Cannot plot a function of 3 or more variables')
        return fig, ax

    def grad(self, acc: int = 1):
        return operators.Gradient(self.grid.nd).apply(field=self, acc=acc)

    def diff(self, axis: int = 0, order: int = 1, acc: int = 1):
        return operators.Diff(order=order, axis=axis).apply(field=self, acc=acc)

    def laplacian(self, acc: int = 1):
        return operators.Laplacian().apply(field=self, acc=acc)
    
    def directional_diff(self, x: tuple, direction: tuple, order: int = 1, acc: int = 1, fd: int = 0):
        '''
        Switching vec -> -vec and fd -> -fd, yields the exact opposite result (for odd order), because the same nodes
        are used for the finite differences, but with opposite signs (for odd order, again). This
        is because for odd order, backward finite difference coefficients are opposite to forward fdc.
        '''
        node = self.grid.node(*x)
        m = findiff.directional_diff_element(grid=self.grid, node=node, direction=direction, order=order, acc=acc, fd=fd)
        return m.dot(self.field.flatten(order='F'))[self.grid.flatten_index(node)]



class VectorField:
    def __init__(self, components: tuple[ScalarField]):
        if len(components) > 1:
            for i in range(len(components)-1):
                components[i]._assert_compatibility(components[i+1])
        
        grid = components[0].grid
        if len(components) > len(grid.shape):
            raise ValueError('ScalarField components more than grid dimensions')
        elif len(components) < len(grid.shape):
            for _ in range(len(grid.shape)-len(components)):
                components += (ScalarField(grid, np.zeros(grid.shape)))
        
        self.grid = grid
        self.nd = grid.nd
        self.x = grid.x
        self.components = components

    def __getitem__(self, i)->ScalarField:
        return self.components[i]
    
    def __add__(self, other: VectorField):
        self._assert_compatibility(other)
        new_fields = [ScalarField(self.grid, self[i]+other[i]) for i in range(self.nd)]
        return VectorField(tuple(new_fields))

    def __sub__(self, other: VectorField):
        self._assert_compatibility(other)
        new_fields = [ScalarField(self.grid, self[i]-other[i]) for i in range(self.nd)]
        return VectorField(tuple(new_fields))
    
    def dot(self, other: VectorField):
        self._assert_compatibility(other)
        scalar_field = self[0]*other[0]
        for i in range(1, self.nd):
            scalar_field += self[i]+other[i]
        scalar_field = ScalarField(self.grid, scalar_field)

    def divergence(self, acc: int = 1):
        return operators.Gradient(self.grid.nd).dot(field=self, acc=acc)

    def curl(self, acc: int = 1):
        return operators.Gradient(self.grid.nd).cross(field=self, acc=acc)

    def _assert_compatibility(self, other: VectorField):
        if self.grid != other.grid:
            raise ValueError('Fields lie on different grids')

def ScalarField_fromfunc(grid: grids.Grid, f):
    return ScalarField(grid=grid, field=f(*grid.x_mesh()))