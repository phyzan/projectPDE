from __future__ import annotations
from .import grids
from . import operators
from . import finitedifferences as findiff
import scipy.interpolate as interp
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class ScalarField:
    '''
    Class responsible for managing discretized scalar fields
    '''
    def __init__(self, grid: grids.Grid, field: np.ndarray):
        '''
        Parameters
        --------------
        grid: The grid object which the field lies on
        field: Numpy array with the same shape as the grid

        '''

        if field.shape != grid.shape:
            raise ValueError('grids.Grid shape and array shape incompatible')
        
        self.grid = grid
        self.field = field
        self.nd = grid.nd
        self.x = grid.x

    def __call__(self, *coords):
        '''
        Evaluate the field at some specific coorfinates.

        e.g.

        def f1(x, y):
            return x**2 + y*np.cos(x-7)

        g = grids.Grid((100, 100), ((0, 10), (0, 10)))
        f2 = ScalarField_fromfunc(g, f1)

        exact = f1(3.5, 8.34)
        approx = f2(3.5, 8.34)

        print(exact, approx)
        '''
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
        '''
        parameters
        ------------
        axis: which axis should the function be evaluated on
        value: The value of the function on that axis

        returns
        -------------
        ScalarField, 1 dimension smaller
        If the field was 1 dimensional, then a single number is returned


        e.g.

        def f1(x, y, z):
            return y*np.cos(y-z) + x**2-z**2

        g = Grid((100, 200, 300), ((0, 10), (0,10), (0, 10)))

        f2 = ScalarField_fromfunc(g, f1)

        #If we want the function evaluated at y = 0,
        #we do as follows
        f3 = f2.evaluate(axis=1, value=0) # This returns f3(x, z) = f2(x, 0, z)
        f3.plot()
        print(f3(4, 1)) # f3(x, z) = x**2 - z**2, so this should be 15
        plt.show()
        '''
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
        '''
        Returns fig, ax created in the plotting
        kwargs are matplotlib parameters passed into ax.plot() (for 1d plots)
        or ax.pcolormesh() (for 2d plots)
        '''
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
        '''
        Computed the gradient of a field


        Parameters
        ----------
        acc: The accuracy of the finite differences used in the differentiation

        Returns
        ------------
        VectorField


        Notes
        ---------
        e.g. For a 3D field, (df/dx, df/dy, df/dz)
        '''
        return operators.Gradient(self.grid.nd).apply(field=self, acc=acc)

    def diff(self, axis: int = 0, order: int = 1, acc: int = 1):
        '''
        Computes partial derivative with repsect to some axis

        parameters
        ------------
        axis: The axis that the partial derivative should be calculated with respect with
        order: Order of differentiation
        acc: Accuracy of finite differences

        Returns
        -------------
        ScalarField
        '''
        return operators.Diff(order=order, axis=axis).apply(field=self, acc=acc)

    def laplacian(self, acc: int = 1):
        '''
        Calculates the laplacian of the field (e.g for a 2D field f(x, y), this is f_xx + f_yy)
        
        Parameters
        ----------
        acc: Accuracy of finite differences

        Returns
        -----------
        ScalarField, that represents the laplacian of the field
        '''
        return operators.Laplacian().apply(field=self, acc=acc)
    
    def directional_diff(self, x: tuple, direction: tuple, order: int = 1, acc: int = 1, fd: int = 0):
        '''
        Computes the directional derivative of a field at given coordinates

        Parameters
        -----------------
        x: coordinates ( e.g x = (9, 12.34) )
        direction: direction used for the differentiation. e.g. direction = (1, 1)
        order: order of differentiation. If order = 1, then we get df/dn. If order = 2, we get d^f/dn^2, where n is the direction
        acc: Accuracy of finite differences
        fd (0, 1 or -1): Central, forward, or backward finite differences
            If fd = 0, central finite differences are used if possible (might not be possible near the grid edges, unless the grid is periodic).
                If not possible backward or forward finite differences will be used. This option does not raises errors.
            If fd = 1, forward finite differences are used if possible
                If not possible, an error will be raised.
            If fd = -1, backward finite differences are used if possible
                If not possible, an error will be raised.

        Returns
        -----------------
        Single number, df/dn evaluated at given coordinates

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