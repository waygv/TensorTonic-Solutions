
#include <cuda_runtime.h>

__global__ void relu(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] = fmaxf(input[tid], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
