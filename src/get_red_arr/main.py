import numpy as np
import os
from ctypes import CDLL, POINTER, c_double, c_int

# 加载共享库
dirnow  = os.path.dirname(os.path.abspath(__file__))
dllfile = os.path.join(dirnow, "get_red_arr.so") 
lib = CDLL(dllfile)

# 定义 C 函数
lib.get_red_arr.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double)]
lib.get_red_arr.restype = None

def raw_get_red_arr(grey_arr: np.ndarray):
    assert isinstance(grey_arr, np.ndarray)
    assert len(grey_arr.shape) == 2
    grey_arr1 = grey_arr.astype(np.double)
    red_arr1  = np.zeros(grey_arr.shape)
    # calling C function
    lib.get_red_arr(
        c_int(grey_arr1.shape[0]), 
        c_int(grey_arr1.shape[1]), 
        grey_arr1.ctypes.data_as(POINTER(c_double)),
        red_arr1.ctypes.data_as(POINTER(c_double)))
    grey_arr2 = grey_arr[::-1,:].astype(np.double) # reverse
    red_arr2 = np.zeros(grey_arr.shape)
    lib.get_red_arr(
        c_int(grey_arr2.shape[0]), 
        c_int(grey_arr2.shape[1]), 
        grey_arr2.ctypes.data_as(POINTER(c_double)),
        red_arr2.ctypes.data_as(POINTER(c_double)))
    red_arr2 = red_arr2[::-1, :] # reverse
    return np.maximum(red_arr1, red_arr2)

def get_red_arr(grey_arr: np.ndarray):
    assert isinstance(grey_arr, np.ndarray)
    assert len(grey_arr.shape) == 2
    n, m = grey_arr.shape
    radius = n//4
    down_half     = grey_arr[:n//2+radius, :]
    up_half       = grey_arr[n//2-radius:, :]
    down_half_ans = np.r_[raw_get_red_arr(down_half)[:-radius,:], np.zeros((n-n//2, m))]
    up_half_ans   = np.r_[np.zeros((n//2, m)), raw_get_red_arr(up_half)[radius:,:]]
    return np.maximum(down_half_ans, up_half_ans)

if __name__ == "__main__": # testing
    # 创建 NumPy 数组
    a = np.array([1, 0, 1, 0, 1, 0], dtype=np.double).reshape((2, 3))
    b = get_red_arr(a)
    print(b)