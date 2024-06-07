all: get_red_arr.so

clean:
	rm *.so

get_red_arr.so: get_red_arr.cu get_red_arr.h
	nvcc -Xcompiler -fPIC -shared -o get_red_arr.so get_red_arr.cu

.PHONY: all clean
