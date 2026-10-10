
#include <cuda_runtime.h>

__global__ void leaky_relu(const float* input, float* output, int N, float alpha) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] = (input[tid] >= 0.0f)
                    ? input[tid]
                    : alpha * input[tid];
    }
}

extern "C" void solve(const float* input, float* output, int N, float alpha) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    leaky_relu<<<blocks, threads>>>(input, output, N, alpha);
    cudaDeviceSynchronize();
}
