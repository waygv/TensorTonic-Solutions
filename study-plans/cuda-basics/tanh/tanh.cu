
#include <cuda_runtime.h>

__global__ void tanh_activation(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] = tanhf(input[tid]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    tanh_activation<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
