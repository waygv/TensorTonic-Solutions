
#include <cuda_runtime.h>

__global__ void sigmoid(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] = 1.0f / (1.0f + expf(-input[tid]));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sigmoid<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
