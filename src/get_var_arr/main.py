import numpy as np
import os
from ctypes import CDLL, POINTER, c_double, c_int

# 加载共享库
dirnow  = os.path.dirname(os.path.abspath(__file__))
dllfile = os.path.join(dirnow, "get_var_arr.so") 
lib = CDLL(dllfile)

# 定义 C 函数
lib.get_var_arr.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double)]
lib.get_var_arr.restype = None

# get var arr
def get_var_arr(raw_arr: np.ndarray):
    assert isinstance(raw_arr, np.ndarray)
    assert len(raw_arr.shape) == 2
    raw_arr = raw_arr.astype(np.double)
    var_arr = np.zeros(raw_arr.shape)
    lib.get_var_arr(
        c_int(raw_arr.shape[0]), 
        c_int(raw_arr.shape[1]), 
        raw_arr.ctypes.data_as(POINTER(c_double)),
        var_arr.ctypes.data_as(POINTER(c_double)))
    return var_arr

if __name__ == "__main__":
    a = np.array([1, 0, 1, 0, 1, 0], dtype=np.double).reshape((2, 3))
    b = get_var_arr(a)
    print(b)