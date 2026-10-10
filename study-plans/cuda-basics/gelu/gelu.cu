
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        float x = input[tid];
        output[tid] = 0.5f * x * (1.0f + erff(x * 0.70710678118f));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (N <= 0) return;

    gelu<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
