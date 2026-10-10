#include <cuda_runtime.h>
#include <math.h>

__global__ void swish(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        float x = input[tid];
        output[tid] = x / (1.0f + expf(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    swish<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}