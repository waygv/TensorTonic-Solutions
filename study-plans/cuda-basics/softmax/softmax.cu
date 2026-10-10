
#include <cuda_runtime.h>

__global__ void softmax(const float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    // Step 1: Find the maximum input value.
    float local_max = -3.402823466e+38F;

    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];

    // Step 2: Sum exp(input[i] - max_val).
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float denominator = sdata[0];

    // Step 3: Normalize every element.
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / denominator;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    softmax<<<1, 256>>>(input, output, N);
    cudaDeviceSynchronize();
}
