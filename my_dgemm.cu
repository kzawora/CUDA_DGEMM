#include "common_header.h"

__global__ void myDgemmKernel_naive(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double beta,
    double* C, int ldc)
{
    int tid_x = blockIdx.x;
    int tid_y = blockIdx.y;
    int mat_c_idx = tid_x + tid_y * ldc;
    C[mat_c_idx] *= beta;
    for (int i = 0; i < k; i++) {
        int mat_a_idx = transa == CUBLAS_OP_T ? tid_x * lda + i : tid_x + i * lda;
        int mat_b_idx = transb == CUBLAS_OP_T ? tid_y + i * ldb : tid_y * ldb + i;
        C[mat_c_idx] += alpha * A[mat_a_idx] * B[mat_b_idx];
    }
}

cudaReturnValue myDgemmHostCode(
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

    dim3 numBlocks(m, n);
    dim3 threadsPerBlock(1);
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
    myDgemmKernel_naive << <numBlocks, threadsPerBlock >> > (
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
