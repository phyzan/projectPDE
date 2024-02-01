from __future__ import annotations
from . import grids
from . import fields
from . import bounds
from . import operators
from . import tools
import sys
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spl
from matplotlib.animation import FuncAnimation
from typing import Callable
import numpy as np
import scipy.sparse as sp
from matplotlib import cm



class PDE:
    '''
    Base class which all Linear Partial Differential Equations inherit from.
    '''
    def __init__(self, grid: grids.Grid, operator: operators.Operator, bcs: bounds.BoundaryConditions, source: Callable[..., float|np.ndarray] = None):
        self.grid = grid
        self.operator = operator
        self.xmesh = self.grid.x_mesh()
        self.bcs = bcs
        self.bcs_filter = bcs.filter_matrix(self.grid)

        if source is None:
            source_arr = np.zeros(self.grid.n_tot)
            def s(t):
                return source_arr
            self.sourcefunc = s
        elif tools._is_timedependent(source):
            def s(t):
                return self.clear_bcs_entries(source(*self.xmesh, t).flatten(order='F'))
            self.sourcefunc = s
        else:
            source_arr = self.clear_bcs_entries(source(*self.xmesh).flatten(order='F'))
            def s(t):
                return source_arr
            self.sourcefunc = s
        
    def source_arr(self, t):
        return self.sourcefunc(t)

    def clear_bcs_entries(self, a: sp.csr_matrix|np.ndarray):
        return self.bcs_filter.dot(a)


class BoundaryValueProblem(PDE):
    def __init__(self, grid: grids.Grid, operator: operators.Operator, bcs: bounds.BoundaryConditions, source: Callable[..., float|np.ndarray] = None):

        super().__init__(grid, operator, bcs, source)

    def bvp_array(self):
        return -self.clear_bcs_entries(self.source_arr(0)) + self.bcs.array(self.grid)

    def bvp_matrix(self, acc: int = 1)->sp.csr_matrix:
        
        oper = self.operator.matrix(grid=self.grid, acc=acc)
        return self.clear_bcs_entries(oper) + self.bcs.matrix(self.grid, acc)
    
    def solve(self, acc: int = 1):
        field = spl.spsolve(self.bvp_matrix(acc=acc), self.bvp_array()).reshape(self.grid.shape, order='F')
        return fields.ScalarField(grid=self.grid, field=field)


class TimeDependentProblem(PDE):
    def __init__(self, grid: grids.Grid, operator: operators.Operator, bcs: bounds.BoundaryConditions, order: int, source: Callable[..., float|np.ndarray] = None):

        self.order = order
        super().__init__(grid, operator, bcs, source)

    def _initial_steps(self, ics: tuple[Callable[..., float|np.ndarray]], dt: float)->np.ndarray[np.ndarray]:
        '''
        ics (tuple): (f(x, y, z), df/dt(x, y, z)...)
        '''
        if len(ics) != self.order:
            raise ValueError('Initial conditions must be as many as the order of the time derivative in the pde')
        
        ics_op = bounds.ics_operator(self.order, dt)
        ics_matrix = np.zeros((self.order, tools.prod(self.grid.shape)))
        for i in range(self.order):
            ics_matrix[i] = ics[i](*self.xmesh).flatten(order='F')
        
        f = ics_op.dot(ics_matrix)
        bvp_matrix = self.clear_bcs_entries(operators.I.matrix(self.grid)) + self.bcs.matrix(self.grid)
        for i in range(self.order):
            bvp_array = self.clear_bcs_entries(f[i]) + self.bcs.array(self.grid, t=i*dt)
            f[i] = spl.spsolve(bvp_matrix, bvp_array)

        return f

    def _solve(self, ics: tuple[Callable[..., float|np.ndarray]], t: float, dt: float, acc: int = 1, frames: int = 0):
        f = self._initial_steps(ics=ics, dt=dt)
        A = self.lhs_matrix(dt, acc)
        B = self.rhs_matrix(dt=dt, acc=acc)
        nt = int(t/dt)+1
        frame_arr = np.zeros((frames, tools.prod(self.grid.shape)))
        k = 0
        for i in range(self.order, nt):
            rhs = sum([B[j].dot(f[j]) for j in range(self.order)]) + dt**self.order*self.source_arr(i*dt) + self.bcs.array(self.grid, i*dt)
            for j in range(self.order-1):
                f[j] = f[j+1]
            f[self.order-1] = spl.spsolve(A, rhs)

            if i*frames > k*nt and k < frames:
                frame_arr[k] = f[-1]
                k += 1
            tools._output_progress(i, nt)

        return fields.ScalarField(grid=self.grid, field=f[-1].reshape(self.grid.shape, order='F')), frame_arr

    def get_frames(self, ics: tuple[Callable[..., float|np.ndarray]], t: float, dt: float, frames: int, acc=1):
        return self._solve(ics=ics, t=t, dt=dt, acc=acc, frames=frames)[1]
    
    def lhs_matrix(self, dt: float, acc: int)->sp.csr_matrix:
        pass
    
    def rhs_matrix(self, dt: float, acc: int)->list[sp.csr_matrix]:
        pass

    def animate(self, ics: tuple[Callable[..., float|np.ndarray]], t: float, dt: float, duration: float, fps: int = 20, acc: int = 1, axes: str = '2d', zlims: tuple[float] = None, cmap: str = None, dpi: int = 200, save: str = ''):
        frames = self.get_frames(ics=ics, t=t, dt=dt, acc=acc, frames=int(fps*duration))
        if self.grid.nd == 1:
            ani = animate_1d(self.grid.x[0], frames, duration, fps, dpi, save)
        elif self.grid.nd == 2:
            ani = animate_2d(*self.grid.x, frames, duration, fps, axes, zlims, cmap, dpi, save)
        else:
            raise ValueError('Cannot animate in higher than 2 dimensions')
        return ani

class FirstOrderTDP(TimeDependentProblem):
    def __init__(self, grid: grids.Grid, operator: operators.Operator, bcs: bounds.BoundaryConditions, source: Callable[..., float|np.ndarray] = None):
        super().__init__(grid, operator, bcs, 1, source)

    def lhs_matrix(self, dt: float, acc: int = 1):
        oper = (1 - dt/2*self.operator).matrix(grid=self.grid, acc=acc)
        return self.clear_bcs_entries(oper) + self.bcs.matrix(self.grid)
    
    def rhs_matrix(self, dt: float, acc: int = 1):
        oper = (1 + dt/2*self.operator).matrix(grid=self.grid, acc=acc)
        return [self.clear_bcs_entries(oper)]

    def solve(self, f0: Callable[..., np.ndarray], t: float, dt: float, acc: int = 1):
        return self._solve(ics=(f0,), t=t, dt=dt, acc=acc, frames=0)[0]
    
class SecondOrderTDP(TimeDependentProblem):
    def __init__(self, grid: grids.Grid, operator: operators.Operator, bcs: bounds.BoundaryConditions, source: Callable[..., float|np.ndarray] = None):
        super().__init__(grid, operator, bcs, 2, source)

    def lhs_matrix(self, dt: float, acc: int = 1):
        return self.clear_bcs_entries(operators.I.matrix(self.grid)) + self.bcs.matrix(self.grid)
    
    def rhs_matrix(self, dt: float, acc: int = 1):
        rhs1 = -self.clear_bcs_entries(operators.I.matrix(self.grid))
        rhs2 = self.clear_bcs_entries((dt**2*self.operator + 2).matrix(grid=self.grid, acc=acc))
        return [rhs1, rhs2]
    
    def solve(self, f0: Callable[..., np.ndarray], df0: Callable[..., np.ndarray], t: float, dt: float, acc: int = 1):
        return self._solve(ics=(f0, df0), t=t, dt=dt, acc=acc, frames=0)[0]



class Poisson(BoundaryValueProblem):
    def __init__(self, grid: grids.Grid, bcs: bounds.BoundaryConditions, source):
        super().__init__(grid=grid, operator=operators.Laplacian(), bcs=bcs, source=source)


class Laplace(BoundaryValueProblem):
    def __init__(self, grid: grids.Grid, bcs: bounds.BoundaryConditions):
        super().__init__(grid=grid, operator=operators.Laplacian(), bcs=bcs, source=None)


class DiffusionEquation(FirstOrderTDP):
    def __init__(self, grid: grids.Grid, bcs: bounds.BoundaryConditions, coefficient: float, source: Callable[..., float|np.ndarray] = None):
        self.D = operators.asoperator(coefficient)
        if tools.isnumeric(coefficient):
            #although this if statement is not needed, the produced operator is more accurate
            #because Diff(order=2) is more accurate than Diff(order=1)**2
            oper = coefficient*operators.Laplacian()
        else:
            grad = operators.Gradient(grid.nd)
            oper = grad * (self.D*grad)
        super().__init__(grid, oper, bcs, source)


class WaveEquation(SecondOrderTDP):
    def __init__(self, grid: grids.Grid, bcs: bounds.BoundaryConditions, c: float|Callable[..., float|np.ndarray], source: Callable[..., float|np.ndarray] = None):
        self.c = operators.asoperator(c)
        oper = self.c**2 * operators.Laplacian()
        super().__init__(grid, oper, bcs, source)



def _process_animation(frames, update, fig, time, fps, dpi, save: str):
    interval = time/(len(frames) - 1)
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    if save == '':
        plt.show()
    else:
        ani.save(sys.path[0]+f'/{save}', fps=fps, dpi=dpi, writer='ffmpeg')
    return ani

def animate_1d(x: np.ndarray, y:np.ndarray, duration: float, fps: float, dpi: int = 200, save=''):
    def update(y):
        line.set_ydata(y)
        return [line]

    fig, ax = plt.subplots()
    line = ax.plot(x, y[0])[0]
    dy = y.max() - y.min()
    ax.set_ylim(y.min()-dy/8, y.max()+dy/8)

    return _process_animation(frames=y, update=update, fig=fig, time=duration, fps=fps, dpi = dpi, save=save)

def animate_2d(x: np.ndarray, y:np.ndarray, f:np.ndarray, duration: float, fps: float, axes: str ='2d', zlims: tuple[float] = None, cmap: str = None, dpi: int = 200, save: str = ''):
    def update(fi: np.ndarray[np.ndarray]):
        ax.clear()
        if axes == '2d':
            ax.pcolormesh(x, y, fi.reshape((y.shape[0], x.shape[0])), norm=norm, shading='auto', cmap=cmap)
            return ax, cbar
        elif axes == '3d':
            ax.plot_surface(X, Y, fi.reshape((x.shape[0], y.shape[0]), order='F'), cmap=cmap, rstride=5, cstride=1, alpha=None)
            ax.set_zlim(*zlims)
            return ax,

    if zlims is None:
        zlims = f.min(), f.max()
    norm = plt.Normalize(*zlims)

    if axes == '2d':
        Lx, Ly = x.max()-x.min(), y.max()-y.min()
        scale = max(Lx, Ly)
        lx = 12.5*Lx/scale
        ly = 10*Ly/scale
        fig, ax = plt.subplots(figsize=(lx, ly))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
    elif axes == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        raise ValueError('2d or 3d are the only accepted parameters for "axes"')  

    return _process_animation(frames=f, update=update, fig=fig, time=duration, fps=fps, dpi=dpi, save=save)
