#include <stdio.h>

// most naive implementation of a matmul in cuda

// A: M x K
// B: K x N
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < N) {

        float total = 0;
        for(int i = 0; i < K; i++) {
            total += A[x * K + i] * B[i * N + y];
        }
        C[x * K + y] = alpha * total + beta * C[x * K + y];
    }
}

const int BLOCKSIZE = 512;

__global__ void sgemm_warp(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C) {
    int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if(x < M && y < N) {

        float total = 0;
        for(int i = 0; i < K; i++) {
            total += A[x * K + i] * B[i * N + y];
        }
        C[x * K + y] = alpha * total + beta * C[x * K + y];
    }
}

// A: M x K
// B: K x N
template<const uint BLOCKSIZE>
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C) {
    int row = blockIdx.x;
    int col = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    int thread_row = threadIdx.x / BLOCKSIZE;
    int thread_col = threadIdx.x % BLOCKSIZE;

    A += row * BLOCKSIZE * K;
    B += BLOCKSIZE * col;
    C += row * BLOCKSIZE * N + BLOCKSIZE * col;

    float tmp = 0;

    for(int blockId; blockId < K; blockId += BLOCKSIZE) {
        As[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        Bs[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];

        __syncthreads();

        for(int j = 0; j < BLOCKSIZE; j++) {
            tmp += As[thread_row * BLOCKSIZE + j] * Bs[j * BLOCKSIZE + thread_col];
        }
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    C[thread_row * K + thread_col] = alpha * tmp + beta * C[thread_row * K + thread_col];
}




// A: M x K
// B: K x N
template<const uint BM, const uint BK, const uint BN, const uint TM>
__global__ void sgemm_shared_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C) {
    int row = blockIdx.x;
    int col = blockIdx.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // warp calculates 32 * TM elements, thread calculates one row of TM elements
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;

    A += row * BM * K;
    B += BN * col;
    C += row * BM * N + BN * col;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;

    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0};

    for(int blockId; blockId < K; blockId += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * N;
        
        for(int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // store the B matrix value that will be used for all of these
            const float tmpB = Bs[dotIdx * BN + threadCol];
            for(int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[((threadRow * TM) + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for(uint resIdx = 0; resIdx < TM; ++resIdx) {
        uint index = (threadRow * TM + resIdx) * N + threadCol;
        C[index] = alpha * threadResults[resIdx] + beta * C[index];
    }

    C[threadRow * K + threadCol] = alpha * tmp + beta * C[threadRow * K + threadCol];
}

// A: M x K
// B: K x N
template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemm_shared_2d_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C) {
    int row = blockIdx.x;
    int col = blockIdx.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int totalNumResults = BM * BN;
    const int numTiles = totalNumResults / (TM * TN);

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);

    A += row * BM * K;
    B += BN * col;
    C += row * BM * N + BN * col;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA = numTiles / BK;

    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB = numTiles / BN;

    float threadResults[TM * TN] = {0.0};

    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(int i = 0; i < K; i += BK) {
        // load memory
        for(int loadPtr = 0; loadPtr < BM; ++loadPtr) {
            As[(innerRowA + loadPtr) * BK + innerColA] = A[(innerRowA + loadPtr) * K + innerColA];
        }

        for(int loadPtr = 0; loadPtr < BK; ++loadPtr) {
            Bs[(innerRowB + loadPtr) * BN + innerColB] = B[(innerRowB + loadPtr) * N + innerColB];
        }
        __syncthreads();

        for(uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for(uint im = 0; im < TM; ++im) {
                regM[im] = 
            }
        }
    }

    for(int i = 0; i < K; i+=BK) {
        
    }
}

