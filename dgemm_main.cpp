#include "common_header.h"
#include <stdlib.h>

void printMatrixColMajor(const double* matrix, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%lf ", matrix[i + j * width]);
        }
        printf("\n");
    }
}

double doubleRand(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}


int syntheticTest(cublasOperation_t transa, cublasOperation_t transb, const int m, const int k, const int n, bool print_matrix = false, const int rand_max = 10) {
    srand(time(NULL));
    const double alpha = rand() % rand_max;
    const double beta = rand() % rand_max;
    const int lda = transa == CUBLAS_OP_N ? m : k;
    const int ldb = transb == CUBLAS_OP_N ? k : n;
    const int ldc = m;

    double* A = (double*)malloc(m * k * sizeof(double));
    for (int i = 0; i < m * k; i++) A[i] = doubleRand(-rand_max, rand_max);
    double* B = (double*)malloc(n * k * sizeof(double));
    for (int i = 0; i < n * k; i++) B[i] = doubleRand(-rand_max, rand_max);
    double* C_cublas = (double*)malloc(m * n * sizeof(double));
    double* C_myKernel = (double*)malloc(m * n * sizeof(double));
    for (int i = 0; i < n * m; i++) {
        double rand_val = doubleRand(-rand_max, rand_max);
        C_cublas[i] = rand_val;
        C_myKernel[i] = rand_val;
    }

    int return_code = 0;

    // run cublasGemm
    cudaReturnValue cublasStatus = cublasDgemmWrapper(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_cublas, ldc);
    if (cublasStatus.status != cudaSuccess) {
        fprintf(stderr, "cuBlasDgemm failed!");
        return_code = 1;
        goto Cleanup;
    }


    // run myKernel
    cudaReturnValue myKernelStatus = myDgemmHostCode(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_myKernel, ldc);
    if (myKernelStatus.status != cudaSuccess) {
        fprintf(stderr, "myKernel failed!");
        return_code = 1;
        goto Cleanup;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return_code = 1;
        goto Cleanup;
    }

    // print results
    printf("cublasDgemm time %lfs\n", cublasStatus.executionTime);
    if (print_matrix) printMatrixColMajor(C_cublas, ldc, n);
    printf("myDgemm time %lfs\n", myKernelStatus.executionTime);
    if (print_matrix) printMatrixColMajor(C_myKernel, ldc, n);

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

Cleanup:
    free(A);
    free(B);
    free(C_myKernel);
    free(C_cublas);

    return return_code;
}

int prettyTest() {
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
    cudaReturnValue cublasStatus = cublasDgemmWrapper(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_cublas, ldc);
    if (cublasStatus.status != cudaSuccess) {
        fprintf(stderr, "cuBlasDgemm failed!");
        return 1;
    }


    // run myKernel
    cudaReturnValue myKernelStatus = myDgemmHostCode(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_myKernel, ldc);
    if (myKernelStatus.status != cudaSuccess) {
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
    printf("cublasDgemm time %lfs, values:\n", cublasStatus.executionTime);
    printMatrixColMajor(C_cublas, ldc, n);
    printf("myDgemm time %lfs, values:\n", myKernelStatus.executionTime);
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

int main() {
    syntheticTest(CUBLAS_OP_T, CUBLAS_OP_T, 2000, 2000, 2000);
}