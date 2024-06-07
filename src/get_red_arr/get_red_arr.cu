#include <cuda_runtime.h>
#include "get_red_arr.h"

#define get_pos(A, i, j) (A[(i) * m + (j)])
#define SQR(x) ((x)*(x))
__global__ void solve_contribution(int base, int len, int nzcnt, int n, int m, int* xpos, int* ypos, float* cuda_grey_arr, float* cuda_red_arr) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x + base;
    if(base <= threadId && threadId < base + len) {
        int j = threadId % m;
        int i = (threadId / m) % n;
        int t = (threadId / m) / n;
        if(0 <= i && i < n && 0 <= j && j < m && 0 <= t && t < nzcnt) {
            int x = xpos[t]; // get position now
            int y = ypos[t];
            float d = sqrtf(SQR(x - i) + SQR(y - j));
            float v = get_pos(cuda_grey_arr, x, y) * Hr / (d + Hr);
            //printf("x = %d, y = %d (%f), t = %d, i = %d, j = %d\n", x, y, get_pos(cuda_grey_arr, x, y), t, i, j);
            if(v > get_pos(cuda_red_arr, i, j)) { // set red color
                atomicExch(&get_pos(cuda_red_arr, i, j), v);
            }
        }
    }
}
#undef SQR

extern "C" { // DLL
void get_red_arr(int n, int m, double* grey_arr, double* red_arr) {
    const auto MATRIX_SIZE = sizeof(float) * n * m;

    float* cuda_grey_arr = nullptr;
    cudaMallocManaged(&cuda_grey_arr, MATRIX_SIZE);
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            get_pos(cuda_grey_arr, i, j) = get_pos(grey_arr, i, j);
        }
    }

    float* cuda_max_value = nullptr;
    cudaMallocManaged(&cuda_max_value, sizeof(float)); 

    assert(n > 0 && m > 0);
    int nzcnt = count_gtr_zero(n, m, grey_arr);
    int* xpos = nullptr;
    int* ypos = nullptr; // save all nz position
    cudaMallocManaged(&xpos, nzcnt * sizeof(int));
    cudaMallocManaged(&ypos, nzcnt * sizeof(int));
    int cnt = 0;
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            if(get_pos(grey_arr, i, j) >= 1.0/256) {
                xpos[cnt] = i;
                ypos[cnt] = j; cnt ++; // move forward
                //printf("x, y = %d, %d\n", xpos[cnt - 1], ypos[cnt - 1]);
            }
        }
    }
    //printf("cnt = %d, nzcnt = %d\n", cnt, nzcnt);

    float* cuda_red_arr = nullptr;
    cudaMallocManaged(&cuda_red_arr, MATRIX_SIZE);
    memset(cuda_red_arr, 0x00, MATRIX_SIZE); // need init

    auto total_cnt = (long long)nzcnt * n * m;
    long long base = 0;
    
    #define min(A, B) (((A)<(B))?(A):(B))
    while(base < total_cnt) {
        int len = min(1024 * 1024, total_cnt - base);
        int blk = (len + 1023) / 1024;
        solve_contribution<<<blk, 1024>>>(base, len, nzcnt, n, m, xpos, ypos, cuda_grey_arr, cuda_red_arr);
        cudaDeviceSynchronize();
        base += len;
    }
    #undef min

    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            get_pos(red_arr, i, j) = get_pos(cuda_red_arr, i, j);
        }
    }
    blur_cpu(n, m, red_arr); // blur
    
    cudaFree(ypos);
    cudaFree(xpos);
    cudaFree(cuda_max_value); // free memory
    cudaFree(cuda_red_arr);
    cudaFree(cuda_grey_arr);
}
}
#undef get_pos