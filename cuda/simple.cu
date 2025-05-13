#include <stdio.h>

__global__ void add(float* a, float* b) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    a[i] = a[i] + b[i];
}

int main() {
    int N = 1 << 10; // 1kb
    float *a, *b, *a_gpu, *b_gpu;
    a = (float*) malloc(sizeof(float) * N);
    b = (float*) malloc(sizeof(float) * N);

    cudaMalloc(&a_gpu, sizeof(float)*N);
    cudaMalloc(&b_gpu, sizeof(float)*N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 1.0003f;
    }

    cudaMemcpy(a_gpu, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    add<<< 1, N >>>(a_gpu, b_gpu);

    cudaMemcpy(a, a_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("[%d]:%.4f ", i, a[i]);
    }

    return 0;
}

