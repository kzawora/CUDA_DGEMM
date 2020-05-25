#include "common_header.h"
#include <stdlib.h>

void printMatrixColMajor(const double* matrix, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%lf\t", matrix[i + j * width]);
        }
        printf("\n");
    }
}

double doubleRand(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

inline void cleanup(double* A, double* B, double* C_myKernel, double* C_cublas) {
    free(A);
    free(B);
    free(C_myKernel);
    free(C_cublas);
}

template <class T>
inline T max(T a, T b) {
    return a > b ? a : b;
}

int syntheticTest(cublasOperation_t transa, cublasOperation_t transb,
    const int m, const int k, const int n,
    bool print_matrix = false, const int rand_max = 10) {
    srand(time(NULL));
    const double alpha = rand() % rand_max;
    const double beta = rand() % rand_max;
    const int lda = max(1, transa == CUBLAS_OP_N ? m : k);
    const int ldb = max(1, transb == CUBLAS_OP_N ? k : n);
    const int ldc = max(1, m);

    double* A = (double*)malloc(m * k * sizeof(double));
    for (int i = 0; i < m * k; i++) A[i] = doubleRand(-rand_max, rand_max);
    double* B = (double*)malloc(n * k * sizeof(double));
    for (int i = 0; i < n * k; i++) B[i] = doubleRand(-rand_max, rand_max);
    double* C_cublas = (double*)malloc(m * n * sizeof(double));
    double* C_myKernelNaive = (double*)malloc(m * n * sizeof(double));
    double* C_myKernelOpt = (double*)malloc(m * n * sizeof(double));

    for (int i = 0; i < n * m; i++) {
        double rand_val = doubleRand(-rand_max, rand_max);
        C_cublas[i] = rand_val;
        C_myKernelNaive[i] = rand_val;
        C_myKernelOpt[i] = rand_val;
    }

    char transa_str[12] = "CUBLAS_OP_N";
    if (transa != CUBLAS_OP_N) transa_str[10] = 'T';
    char transb_str[12] = "CUBLAS_OP_N";
    if (transb != CUBLAS_OP_N) transb_str[10] = 'T';

    printf("TEST: M=%d, K=%d N=%d, transa=%s, transb=%s\n", m, k, n, transa_str, transb_str);

    int return_code = 0;
    // run cublasGemm
    cudaReturnValue cublasStatus = cublasDgemmWrapper(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_cublas, ldc);
    if (cublasStatus.status != cudaSuccess) {
        fprintf(stderr, "cuBlasDgemm failed!");
        return_code = 1;
        cleanup(A, B, C_myKernelNaive, C_cublas);
        return return_code;
    }


    // run myKernel
    cudaReturnValue myKernelStatusNaive = myDgemmHostCodeNaive(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_myKernelNaive, ldc);
    if (myKernelStatusNaive.status != cudaSuccess) {
        fprintf(stderr, "myKernel failed!");
        return_code = 1;
        cleanup(A, B, C_myKernelNaive, C_cublas);
        return return_code;
    }

    // run myKernel
    cudaReturnValue myKernelStatusOpt = myDgemmHostCodeOpt(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_myKernelOpt, ldc);
    if (myKernelStatusOpt.status != cudaSuccess) {
        fprintf(stderr, "myKernel failed!");
        return_code = 1;
        cleanup(A, B, C_myKernelNaive, C_cublas);
        return return_code;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return_code = 1;
        cleanup(A, B, C_myKernelNaive, C_cublas);
        return return_code;
    }

    // print results
    printf("cublasDgemm time %lfs\n", cublasStatus.executionTime);
    if (print_matrix) printMatrixColMajor(C_cublas, ldc, n);
    printf("myDgemmNaive time %lfs\n", myKernelStatusNaive.executionTime);
    if (print_matrix) printMatrixColMajor(C_myKernelNaive, ldc, n);
    printf("myDgemmOpt time %lfs\n", myKernelStatusOpt.executionTime);
    if (print_matrix) printMatrixColMajor(C_myKernelOpt, ldc, n);

    // compare results
    int errorCounter = 0;
    double epsilon = 1e-9;
    for (int i = 0; i < ldc; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C_myKernelOpt[i + j * ldc] - C_cublas[i + j * ldc]) > epsilon) {
                errorCounter++;
                printf("Value mismatch at (%d,%d):\n  Expected: %lf\n  Actual: %lf\n", i, j, C_cublas[i + j * ldc], C_myKernelOpt[i + j * ldc]);
            }
        }
    }
    if (errorCounter == 0)
        printf("[PASSED] No mismatches found.\n\n");
    else
        printf("[FAILED] %d mismatch(es) found.\n\n", errorCounter);

    cleanup(A, B, C_myKernelNaive, C_cublas);
    free(C_myKernelOpt);
    return return_code;
}

int prettyTest() {
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;
    const int m = 2;
    const int n = 4;
    const int k = 3;
    const double alpha = 0.5;
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

    double C_cublas[m * n] = {
        1.,1.,1., 1.,
        1.,1.,1., 1.
    };

    double C_myKernel[m * n] = {
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
    cudaReturnValue myKernelStatus = myDgemmHostCodeNaive(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C_myKernel, ldc);
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
///    syntheticTest(CUBLAS_OP_T, CUBLAS_OP_T, 16, 16, 16);
///    syntheticTest(CUBLAS_OP_N, CUBLAS_OP_T, 16, 32, 64);
///    syntheticTest(CUBLAS_OP_T, CUBLAS_OP_N, 64, 256, 256);
///    syntheticTest(CUBLAS_OP_N, CUBLAS_OP_N, 17, 257, 75);
///    syntheticTest(CUBLAS_OP_N, CUBLAS_OP_T, 1234, 2345, 1230);
    syntheticTest(CUBLAS_OP_N, CUBLAS_OP_N, 2048, 2048, 2048);
//    syntheticTest(CUBLAS_OP_N, CUBLAS_OP_T, 1024, 1024, 1024);
}