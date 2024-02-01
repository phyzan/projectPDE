from __future__ import annotations
from typing import Callable
import itertools
import numpy as np


class Metric:
    '''time independent metric g_ij'''
    def __init__(self, func: Callable[..., np.ndarray], nd: int):
        '''
        User should ensure the metric is symmetric for now
        '''
        self.nd = nd
        self.lv_tensor = levi_cevita_tensor(self.nd)
        self.func = func

    def __call__(self, *coords):
        return self.func(*coords)

    def inv(self, *coords):
        g = self(*coords)
        return np.linalg.inv(g)
    
    def det(self, *coords):
        return np.linalg.det(self(*coords))
    
    def __eq__(self, other: Metric):
        return self.func == other.func and self.nd == other.nd
    

class DiagonalMetric(Metric):

    def __init__(self, diag: Callable[..., np.ndarray], nd):

        def func(*coords):
            return np.diag(diag(*coords))
        
        self.diag = diag
        super().__init__(func=func, nd=nd)
    
    def det(self, *coords):
        return np.prod(self.diag(*coords))


class CartesianMetric(DiagonalMetric):
    def __init__(self, nd):
        self._metric = np.eye(nd)

        def diag(*coords):
            return np.ones(nd)

        super().__init__(diag, nd)

    def __call__(self, *coords):
        return self._metric
    
    def __eq__(self, other: Metric):
        return isinstance(other, CartesianMetric) and other.nd == self.nd
    
    def inv(self, *coords):
        return self._metric

    def det(self, *coords):
        return 1


class Tensor:
    def __init__(self, func: Callable[..., np.ndarray], rank: int, metric: Metric):
        self.func = func
        self.rank = rank
        self.metric = metric
        self.types = np.ones(rank, dtype=int) # all upper indices

    def __call__(self, *coords):
        return self.func(*coords)
    
class Vector(Tensor):
    def __init__(self, func: Callable[..., np.ndarray], metric: Metric):
        super().__init__(func=func, rank=1, metric=metric)

    def dual(self, *coords):
        return self.metric(*coords).dot(self(*coords))
    
    def norm(self, *coords):
        return self(*coords).dot(self.dual(*coords))
    

def levi_cevita_tensor(dim):   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr


def normal_to(vecs:list[np.ndarray], metric: np.ndarray):
    '''
    e.g.

    vecs = [1, 2, 3] and [6, 0, 9]
    metric = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
    '''
    nd = len(vecs) + 1 #==len(vecs[i]) for all i
    dualvecs = [metric.dot(vec) for vec in vecs]
    res = levi_cevita_tensor(nd)
    for i in range(nd-1):
        res = np.tensordot(res, dualvecs[i], axes=([1], [0]))
    return res