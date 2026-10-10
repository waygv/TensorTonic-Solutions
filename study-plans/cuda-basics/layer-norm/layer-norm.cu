#include <cuda_runtime.h>

__global__ void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int M,
    int N,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Each thread processes some elements of this row.
    for (int j = tid; j < N; j += blockDim.x) {
        float x = input[row * N + j];
        sum += x;
        sum_sq += x * x;
    }

    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    // Reduce the partial sums across the block.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / N;
    float variance = s_sum_sq[0] / N - mean * mean;

    // Protect against tiny negative variance from rounding.
    variance = fmaxf(variance, 0.0f);

    float inv_std = rsqrtf(variance + eps);

    // Normalize and apply the affine transformation.
    for (int j = tid; j < N; j += blockDim.x) {
        float x = input[row * N + j];
        float normalized = (x - mean) * inv_std;
        output[row * N + j] = normalized * gamma[j] + beta[j];
    }
}

extern "C" void solve(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int M,
    int N,
    float eps
) {
    if (M <= 0 || N <= 0) return;

    int threads = 256;

    layer_norm<<<M, threads>>>(
        input, gamma, beta, output, M, N, eps
    );

    cudaDeviceSynchronize();
}