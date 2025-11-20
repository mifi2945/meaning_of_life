########################
# This file demonstrates CPU parallelization using the numba library
########################

from numba import njit, prange
import numpy as np
import time


@njit(parallel=True)
def prange_test(A):
    s = 0
    for i in prange(A.shape[0]):
        s += A[i]
    return s


def regular_func(A):
    s = 0
    for i in range(A.shape[0]):
        s += A[i]
    return s

if __name__ == '__main__':
    # set N for different sizes
    N = 200
    prange_test(np.ones(N))

    start = time.perf_counter()
    regular_func(np.ones(N))
    end = time.perf_counter()
    r_time = end-start
    print(f'Time for python (for sufficiently large dimensions): {r_time}')

    start = time.perf_counter()
    prange_test(np.ones(N))
    end = time.perf_counter()
    n_time = end-start
    print(f'Time for numba (for sufficiently large dimensions): {n_time}')

    print(f'Numba \033[0;36mwas {'' if n_time < r_time else 'not '}\033[0mfaster than regular python.')