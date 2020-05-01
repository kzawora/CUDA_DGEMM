#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <stdio.h>

cudaError_t cublasDgemmWrapper(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc);

cudaError_t myDgemmHostCode(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc);

__global__ void myDgemmKernel_naive(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double beta,
    double* C, int ldc)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int mat_c_idx = tid_x + tid_y * ldc;
    C[mat_c_idx] *= beta;
    for (int i = 0; i < k; i++) {
        int mat_a_idx = transa == CUBLAS_OP_T ? tid_x * lda + i : tid_x + i * lda;
        int mat_b_idx = transb == CUBLAS_OP_T ? tid_y + i * ldb : tid_y * ldb + i;
        C[mat_c_idx] += A[mat_a_idx] * B[mat_b_idx];
    }
}

void printMatrixColMajor(const double* matrix, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%lf ", matrix[i + j * width]);
        }
        printf("\n");
    }
}

int main() {
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;
    const int m = 2;
    const int n = 4;
    const int k = 3;
    const double alpha = 1.;
    const double beta = 2.;
    const int lda = transa == CUBLAS_OP_N ? m : k;
    const int ldb = transb == CUBLAS_OP_N ? k : n;
    const int ldc = m;

    const double A[m * k] = {
        1.,2.,3.,
        4.,5.,6.
    };

    const double B[n * k] = {
        1.,2.,3., 4.,
        5.,6.,7.,8.,
        9.,10.,11.,12.
    };

    double C_myKernel[m * n] = {
        1.,1.,1., 1.,
        1.,1.,1., 1.
    };
    double C_cublas[m * n] = {
        1.,1.,1., 1.,
        1.,1.,1., 1.
    };

    // run cublasGemm
    cudaError_t cublasStatus = cublasDgemmWrapper(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_cublas, ldc);
    if (cublasStatus != cudaSuccess) {
        fprintf(stderr, "cuBlasDgemm failed!");
        return 1;
    }


    // run myKernel
    cudaError_t myKernelStatus = myDgemmHostCode(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_myKernel, ldc);
    if (myKernelStatus != cudaSuccess) {
        fprintf(stderr, "myKernel failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // print results
    printf("cublasDgemm:\n");
    printMatrixColMajor(C_cublas, ldc, n);
    printf("myKernel:\n");
    printMatrixColMajor(C_myKernel, ldc, n);

    // compare results
    int errorCounter = 0;
    double epsilon = 1e-9;
    for (int i = 0; i < ldc; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C_myKernel[i + j * ldc] - C_cublas[i + j * ldc]) > epsilon) {
                errorCounter++;
                printf("Value mismatch at (%d,%d):\n  Expected: %lf\n  Actual: %lf\n", i, j, C_cublas[i + j * ldc], C_myKernel[i + j * ldc]);
            }
        }
    }
    if (errorCounter == 0)
        printf("No mismatches found.\n");
    else
        printf("%d mismatch(es) found.\n", errorCounter);

    return 0;
}


cudaError_t myDgemmHostCode(
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
    cublasHandle_t handle;
    cublasStatus_t stat;

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

    dim3 threadsPerBlock(m, n);
    dim3 numBlocks(1);
    // Launch a kernel on the GPU with one thread for each element.
    myDgemmKernel_naive<<<numBlocks, threadsPerBlock>>> (
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

    return cudaStatus;
}


cudaError_t cublasDgemmWrapper(
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
    cublasHandle_t handle;
    cublasStatus_t stat;

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


    stat = cublasDgemm(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        dev_A, lda,
        dev_B, ldb,
        beta,
        dev_C, ldc
    );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cublasDgemm launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

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

    return cudaStatus;
}
