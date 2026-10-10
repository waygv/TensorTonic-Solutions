#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int N,
    int D
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        float sum = 0.0f;

        for (int d = 0; d < D; d++) {
            sum += Q[i * D + d] * K[j * D + d];
        }

        float scale = rsqrtf((float)D);
        scores[i * N + j] = sum * scale;
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_data[256];

    // Step 1: Find the maximum score in this row.
    float local_max = -FLT_MAX;

    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * N + j]);
    }

    s_data[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] = fmaxf(
                s_data[tid],
                s_data[tid + stride]
            );
        }
        __syncthreads();
    }

    float row_max = s_data[0];

    // Step 2: Exponentiate shifted scores and calculate partial sums.
    float local_sum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(scores[row * N + j] - row_max);
        scores[row * N + j] = e;
        local_sum += e;
    }

    s_data[tid] = local_sum;
    __syncthreads();

    // Step 3: Reduce partial sums.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = s_data[0];

    // Step 4: Normalize into probabilities.
    for (int j = tid; j < N; j += blockDim.x) {
        scores[row * N + j] /= row_sum;
    }
}

__global__ void av_kernel(
    const float* attn,
    const float* V,
    float* output,
    int N,
    int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && d < D) {
        float sum = 0.0f;

        for (int j = 0; j < N; j++) {
            sum += attn[i * N + j] * V[j * D + d];
        }

        output[i * D + d] = sum;
    }
}

extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int N,
    int D
) {
    if (N <= 0 || D <= 0) return;

    float* scores;
    cudaMalloc((void**)&scores, (size_t)N * N * sizeof(float));

    // Stage 1: Q * K^T / sqrt(D)
    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);

    scores_kernel<<<sBlocks, sThreads>>>(
        Q, K, scores, N, D
    );

    // Stage 2: Stable softmax across each row.
    softmax_rows_kernel<<<N, 256>>>(scores, N);

    // Stage 3: Attention weights * V
    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);

    av_kernel<<<oBlocks, oThreads>>>(
        scores, V, output, N, D
    );

    cudaDeviceSynchronize();
    cudaFree(scores);
}