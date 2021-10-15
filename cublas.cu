#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
/* #define M 6 */
/* #define N 5 */
/* #define IDX2C(i,j,ld) (((j)*(ld))+(i)) */


unsigned long timestampMicros() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  unsigned long time_in_micros = 1000000 * tv.tv_sec + tv.tv_usec;
  return time_in_micros;
}


int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    if (argc != 4) {
      printf("./multblas m n k");
      return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    clock_t tStart = clock();

    cublasHandle_t handle;
    cublasCreate(&handle);

    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c;
    cudaMallocHost((void **) &h_a, sizeof(float)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(float)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(float)*m*k);

    clock_t tAlloc = clock();
    printf("created handle, alloc on device: %fs\n", ((double)tAlloc - tStart)/CLOCKS_PER_SEC);

    // random initialize matrix A
    //printf("\n\nA\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          h_a[i * n + j] = rand()/(RAND_MAX*2.0f)-1.0;
            //printf("%.2f ", h_a[i * n + j]);
        }
        //printf("\n");
    }

    // random initialize matrix B
    //printf("\n\nB\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
          h_b[i * k + j] = rand()/(RAND_MAX*2.0f)-1.0;
            //printf("%.2f ", h_b[i * k + j]);
        }
        //printf("\n");
    }

    // random initialize matrix C
    //printf("\n\nC\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
          h_c[i * k + j] = rand()/(RAND_MAX*2.0f)-1.0;
          //printf("%.2f ", h_c[i * k + j]);
        }
        //printf("\n");
    }

    clock_t tInit = clock();
    printf("random values: %fs\n", ((double)tInit - tAlloc)/CLOCKS_PER_SEC);

    /* float gpu_elapsed_time_ms; */

    // some events to count the execution time
    /* cudaEvent_t start, stop; */
    /* cudaEventCreate(&start); */
    /* cudaEventCreate(&stop); */

    // start to count execution time of GPU version
    /* cudaEventRecord(start, 0); */
    // Allocate memory space on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*m*n);
    cudaMalloc((void **) &d_b, sizeof(float)*n*k);
    cudaMalloc((void **) &d_c, sizeof(float)*m*k);

    // copy matrix A, B, C from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(float)*m*k, cudaMemcpyHostToDevice);

    clock_t tCopy = clock();
    printf("copy to device: %fs\n", ((double)tCopy-tInit)/CLOCKS_PER_SEC);


    /* unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE; */
    /* unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE; */
    /* dim3 dimGrid(grid_cols, grid_rows); */
    /* dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); */

    // Launch kernel
    // Result is left in C
    float identity = 1.0f;
    float* alpha = &identity;
    float* beta = &identity;
    auto result = cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                alpha,
                d_a, m,
                d_b, n,
                beta,
                d_c, m
                );
    if (result != 0) {
      printf("FAILED: %d", result);
    }

    clock_t tGemm = clock();
    printf("gemm: %fs\n", ((double)tGemm-tCopy)/CLOCKS_PER_SEC);

    /* auto gemmUs = timestampMicros(); */
    /* printf("gemm: %ldus", gemmUs-cpUs); */

    // Transefr results from device to host
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);

    clock_t tRet = clock();
    printf("copied back to host: %fs\n", ((double)tRet-tGemm)/CLOCKS_PER_SEC);
    //printf("copied back to host: %ldus", timestampMicros()-gemmUs);

    //    cudaThreadSynchronize();
    // time counting terminate
    /* cudaEventRecord(stop, 0); */
    /* cudaEventSynchronize(stop); */

    // compute time elapse on GPU computing
    /* cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop); */
    /* printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms); */

    /* //printf("\n\nResult\n"); */
    /* for (int i = 0; i < m; ++i) { */
    /*     for (int j = 0; j < k; ++j) { */
    /*       //printf("%.2f ", h_c[i * k + j]); */
    /*     } */
    /*     //printf("\n"); */
    /* } */


    cublasDestroy(handle);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}
