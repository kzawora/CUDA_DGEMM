#include "common_header.h"
#define TILE_SIZE 32
#define TILE_M TILE_SIZE
#define TILE_N TILE_SIZE
#define SELECTED_KERNEL myDgemmKernel_opt_32x32

__global__ void myDgemmKernel_opt_shared_mem_swizzling(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double beta,
    double* C, int ldc)
{
    __shared__ double mat_a_shared_tile[TILE_SIZE * TILE_SIZE];
    __shared__ double mat_b_shared_tile[TILE_SIZE * TILE_SIZE];

    // get initial result value from global mem
    int m_read_offset = blockIdx.x * TILE_SIZE + threadIdx.x;
    int n_read_offset = blockIdx.y * TILE_SIZE + threadIdx.y;
    int mat_c_write_offset = m_read_offset + n_read_offset * ldc;
    double result = C[mat_c_write_offset] * beta;

    for (int k_iter = 0; k_iter < k / TILE_SIZE; k_iter++) {
        int k_iter_base_offset = k_iter * TILE_SIZE;

        // matrix A&B write offsets for shared memory
        int mat_a_shared_write_offset = threadIdx.x * TILE_SIZE + threadIdx.y;
        int mat_b_shared_write_offset = threadIdx.y * TILE_SIZE + threadIdx.x;

        // matrix A&B read offsets for global memory
        int mat_a_global_read_offset =
            (k_iter_base_offset + threadIdx.x) * lda + (blockIdx.x * TILE_SIZE + threadIdx.y);
        int mat_b_global_read_offset =
            (k_iter_base_offset + threadIdx.x) * ldb + (blockIdx.y * TILE_SIZE + threadIdx.y);

        // copy global mem tiles to shared mem
        mat_a_shared_tile[mat_a_shared_write_offset] = A[mat_a_global_read_offset];
        mat_b_shared_tile[mat_b_shared_write_offset] = B[mat_b_global_read_offset];

        // sync after copy
        __syncthreads();

        // dot product loop
#pragma unroll
        for (int dp_iter = 0; dp_iter < TILE_SIZE; dp_iter++) {
            int mat_a_read_offset = transa == CUBLAS_OP_T ?
                threadIdx.x * TILE_SIZE + dp_iter :
                threadIdx.x + TILE_SIZE * dp_iter;
            int mat_b_read_offset = transb == CUBLAS_OP_T ?
                threadIdx.y * TILE_SIZE + dp_iter :
                threadIdx.y + TILE_SIZE * dp_iter;
            result += alpha * mat_a_shared_tile[mat_a_read_offset] * mat_b_shared_tile[mat_b_read_offset];
        }

        // sync to make sure shared mem can be overwritten
        __syncthreads();
    }
    // save result to global mem
    C[mat_c_write_offset] = result;
}

__global__ void myDgemmKernel_32x32_shared_mem(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double beta,
    double* C, int ldc)
{
    __shared__ double mat_a_shared_tile[TILE_SIZE * TILE_SIZE];
    __shared__ double mat_b_shared_tile[TILE_SIZE * TILE_SIZE];

    // get initial result value from global mem
    int m_read_offset = blockIdx.x * TILE_SIZE + threadIdx.x;
    int n_read_offset = blockIdx.y * TILE_SIZE + threadIdx.y;
    int mat_c_write_offset = m_read_offset + n_read_offset * ldc;
    double result = C[mat_c_write_offset] * beta;

    for (int k_iter = 0; k_iter < k / TILE_SIZE; k_iter++) {
        int k_iter_base_offset = k_iter * TILE_SIZE;

        // matrix A&B write offsets for shared memory
        int mat_a_shared_write_offset = threadIdx.x * TILE_SIZE + threadIdx.y;
        int mat_b_shared_write_offset = threadIdx.y * TILE_SIZE + threadIdx.x;

        // matrix A&B read offsets for global memory
        int mat_a_global_read_offset = transa == CUBLAS_OP_T ?
            (k_iter_base_offset + threadIdx.x) + lda * (blockIdx.x * TILE_SIZE + threadIdx.y) :
            (k_iter_base_offset + threadIdx.x) * lda + (blockIdx.x * TILE_SIZE + threadIdx.y);
        int mat_b_global_read_offset = transb == CUBLAS_OP_T ?
            (k_iter_base_offset + threadIdx.x) * ldb + (blockIdx.y * TILE_SIZE + threadIdx.y) :
            (k_iter_base_offset + threadIdx.x) + ldb * (blockIdx.y * TILE_SIZE + threadIdx.y);

        // copy global mem tiles to shared mem
        mat_a_shared_tile[mat_a_shared_write_offset] = A[mat_a_global_read_offset];
        mat_b_shared_tile[mat_b_shared_write_offset] = B[mat_b_global_read_offset];

        // sync after copy
        __syncthreads();

        // dot product loop
#pragma unroll
        for (int dp_iter = 0; dp_iter < TILE_SIZE; dp_iter++) {
            int mat_a_read_offset = threadIdx.x + TILE_SIZE * dp_iter;
            int mat_b_read_offset = threadIdx.y * TILE_SIZE + dp_iter;
            result += alpha * mat_a_shared_tile[mat_a_read_offset] * mat_b_shared_tile[mat_b_read_offset];
        }

        // sync to make sure shared mem can be overwritten
        __syncthreads();
    }
    // save result to global mem
    C[mat_c_write_offset] = result;
}


__global__ void myDgemmKernel_opt_32x32(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double beta,
    double* C, int ldc)
{
    int m_read_offset = blockIdx.x * TILE_M + threadIdx.x;
    int n_read_offset = blockIdx.y * TILE_N + threadIdx.y;
    int mat_c_write_offset = m_read_offset + n_read_offset * ldc;

    double result = C[mat_c_write_offset] * beta;
    for (int k_iter = 0; k_iter < k; k_iter++) {
        int mat_a_read_offset = transa == CUBLAS_OP_T ?
            m_read_offset * lda + k_iter :
            m_read_offset + lda * k_iter;
        int mat_b_read_offset = transb == CUBLAS_OP_T ?
            n_read_offset + ldb * k_iter :
            n_read_offset * ldb + k_iter;
        result += alpha * A[mat_a_read_offset] * B[mat_b_read_offset];
    }
    C[mat_c_write_offset] = result;
}

cudaReturnValue myDgemmHostCodeOpt(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc) {

    double* dev_A = 0;
    const int dev_A_size = m * k * sizeof(double);
    double* dev_B = 0;
    const int dev_B_size = n * k * sizeof(double);
    double* dev_C = 0;
    const int dev_C_size = m * n * sizeof(double);

    cudaError_t cudaStatus;
    double executionTime = -1.;
    cublasHandle_t handle;
    cublasStatus_t stat;

    dim3 numBlocks(m / TILE_M, n / TILE_N);
    dim3 threadsPerBlock(TILE_M, TILE_N);
    clock_t t;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)& dev_A, dev_A_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& dev_B, dev_B_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& dev_C, dev_C_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        cudaStatus = cudaErrorNotSupported;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, dev_A_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_B, B, dev_B_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_C, C, dev_C_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // start time measurement
    t = clock();
    // Launch a kernel on the GPU with one thread for each element.
    SELECTED_KERNEL << <numBlocks, threadsPerBlock >> > (
        transa, transb,
        m, n, k,
        *alpha,
        dev_A, lda,
        dev_B, ldb,
        *beta,
        dev_C, ldc
        );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // measure time
    t = clock() - t;
    executionTime = ((double)t) / CLOCKS_PER_SEC;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, dev_C, dev_C_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return { cudaStatus, executionTime };
}
