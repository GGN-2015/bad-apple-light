all: get_var_arr.so

clean:
	rm *.so

get_var_arr.so: get_var_arr.cu
	nvcc -Xcompiler -fPIC -shared -o get_var_arr.so get_var_arr.cu

.PHONY: all clean
