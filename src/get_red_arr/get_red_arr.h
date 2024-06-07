#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define get_pos(A, i, j) (A[(i) * m + (j)])

int count_gtr_zero(int n, int m, double* grey_arr) {
    int ans = 0;
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            if(get_pos(grey_arr, i, j) >= 1.0/256) {
                ans += 1;
            }
        }
    }
    return ans;
}

double sqr(double x) {
    return x * x;
}

const double Hr = 10;

double getcolor(int x, int y, int i, int j) {
    double d = sqrt(sqr(x - i) + sqr(y - j));
    return Hr / (d + Hr);
}

void blur_cpu(int n, int m, double* red_arr) {
    double* new_arr = (double*)malloc(sizeof(double) * n * m);
    memcpy(new_arr, red_arr, sizeof(double) * n * m);
    int radius = (double)n / 500 * 8;
    if(radius <= 0) { // min radius
        radius = 1;
    }
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            double sum = 0;
            int cnt = 0;
            for(int dx = -radius; dx < radius; dx += 1) {
                for(int dy = -radius; dy < radius; dy += 1) {
                    if(0 <= i + dx && i + dx < n && 0 <= j + dy && j + dy < m) {
                        sum += get_pos(red_arr, i + dx, j + dy);
                        cnt += 1;
                    }
                }
            }
            get_pos(new_arr, i, j) = sum / cnt; // blur
        }
    }
    memcpy(red_arr, new_arr, sizeof(double) * n * m);
    free(new_arr);
}

void get_red_arr_cpu(int n, int m, double* grey_arr, double* red_arr) {
    assert(n > 0 && m > 0);
    int nzcnt = count_gtr_zero(n, m, grey_arr);
    int* xpos = (int*)malloc(nzcnt * sizeof(int));
    int* ypos = (int*)malloc(nzcnt * sizeof(int)); // save all nz position
    int cnt = 0;
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < m; j += 1) {
            if(get_pos(grey_arr, i, j) > 0) {
                xpos[cnt] = i;
                ypos[cnt] = j; cnt ++; // move forward
            }
        }
    }
    memset(red_arr, 0x00, sizeof(double) * n * m);
    for(int t = 0; t < nzcnt; t += 1) {
        int x = xpos[t];
        int y = ypos[t];
        for(int i = 0; i < n; i += 1) {
            for(int j = 0; j < m; j += 1) {
                double old_v    = get_pos(red_arr, i, j);
                double grey_col = get_pos(grey_arr, x, y);
                double new_v    = grey_col * getcolor(x, y, i, j);
                if(new_v > old_v) {
                    get_pos(red_arr, i, j) = new_v;
                }
            }
        }
    }
    blur_cpu(n, m, red_arr);
    free(xpos);
    free(ypos);
}
#undef get_pos