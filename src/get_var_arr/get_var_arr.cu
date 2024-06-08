#include <assert.h>
#include <cuda_runtime.h>

#define get_pos(A, i, j) (A[(i) * m + (j)])
#define SQR(x) ((x)*(x))
__global__ void solve_contribution(int base, int len, int n, int m, float* cuda_raw_arr, float* cuda_var_arr) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x + base;
    if(base <= threadId && threadId < base + len) {
        int j = threadId % m;
        int i = threadId / m;
        if(0 <= i && i < n && 0 <= j && j < m) {
            const int radius = 2;
            float sum = 0;
            float sqr = 0;
            int   cnt = 0;
            for(int dx = -radius; dx <= radius; dx += 1) {
                for(int dy = -radius; dy <= radius; dy += 1) {
                    int nx = i + dx;
                    int ny = j + dy;
                    if(0 <= nx && nx < n && 0 <= ny && ny < m) {
                        sum += get_pos(cuda_raw_arr, nx, ny);
                        sqr += SQR(get_pos(cuda_raw_arr, nx, ny));
                        cnt += 1;
                    }
                }
            }
            get_pos(cuda_var_arr, i, j) = (sqr / cnt) - SQR(sum / cnt);
            if(get_pos(cuda_var_arr, i, j) <= 0) { // keep positive
                get_pos(cuda_var_arr, i, j) = 0;
            }
        }
    }
}
#undef SQR

extern "C" { // DLL
void get_var_arr(int n, int m, double* raw_arr, double* ans_arr) {
    assert(n > 0 && m > 0);
    const auto MATRIX_SIZE = sizeof(float) * n * m;

    float* cuda_raw_arr = nullptr;
    cudaMallocManaged(&cuda_raw_arr, MATRIX_SIZE);
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            get_pos(cuda_raw_arr, i, j) = get_pos(raw_arr, i, j);
        }
    }

    float* cuda_ans_arr = nullptr;
    cudaMallocManaged(&cuda_ans_arr, MATRIX_SIZE);

    long long base = 0;
    long long total_cnt = (long long) n * m;
    #define min(A, B) (((A)<(B))?(A):(B))
    while(base < total_cnt) {
        int len = min(1024 * 1024, total_cnt - base);
        int blk = (len + 1023) / 1024;
        solve_contribution<<<blk, 1024>>>(base, len, n, m, cuda_raw_arr, cuda_ans_arr);
        cudaDeviceSynchronize();
        base += len;
    }
    #undef min

    for(int i = 0; i < n; i += 1) { // copy back
        for(int j = 0; j < m; j += 1) {
            get_pos(ans_arr, i, j) = get_pos(cuda_ans_arr, i, j);
        }
    }
    
    cudaFree(cuda_ans_arr); // free memory
    cudaFree(cuda_raw_arr);
}
}
#undef get_pos