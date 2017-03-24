/**
 * @file cublas_gemm.cpp
 * @author Ryan Curtin
 *
 * Perform a GEMM call with square matrices of the size given as argv[1].
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <armadillo>

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "what is the size of the matrix you want to multiply!?" <<
        std::endl;
    return -1;
  }

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int size = atoi(argv[1]);

  // Allocate random matrix.
  arma::wall_clock clock;
  clock.tic();
  arma::mat x(size, size, arma::fill::randu);
  arma::mat y(size, size, arma::fill::randu);
  arma::mat z(size, size); // Results.
  std::cout << "matrix initialization time: " << clock.toc() << "s\n";

  const size_t bufSize = size * size * sizeof(double);

  double* devPtrA;
  double* devPtrB;
  double* devPtrC;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout << "CUBLAS initialization failed\n";
    return EXIT_FAILURE;
  }

  cudaStat = cudaMalloc((void**) &devPtrA, bufSize);
  cudaStat = cudaMalloc((void**) &devPtrB, bufSize);
  cudaStat = cudaMalloc((void**) &devPtrC, bufSize);
  if (cudaStat != cudaSuccess)
  {
    std::cout << "device memory allocation failed\n";
    return EXIT_FAILURE;
  }

  clock.tic();
  cudaStat = cudaMemcpy(devPtrA, x.memptr(), bufSize, cudaMemcpyHostToDevice);
  cudaStat = cudaMemcpy(devPtrB, y.memptr(), bufSize, cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess)
  {
    std::cout << "data load to GPU failed\n";
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  std::cout << "buffer copy: " << clock.toc() << "s\n";

  // Do matrix multiply.
  clock.tic();
  double alpha = 1.0;
  double beta = 0.0;
  stat = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,
      devPtrA, size, devPtrB, size, &beta, devPtrC, size);
  // Make sure multiplication finishes.
  cudaThreadSynchronize();
  std::cout << "multiply time: " << clock.toc() << "s\n";
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout << "failed multiply\n";
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  stat = cublasGetMatrix(size, size, sizeof(double), devPtrC, size, z.memptr(),
      size);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout << "data load from GPU failed\n";
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  cudaFree(devPtrA);
  cudaFree(devPtrB);
  cudaFree(devPtrC);
  cublasDestroy(handle);

  // Check correctness.
  clock.tic();
  arma::mat zz = x * y;
  std::cout << "cpu multiply time: " << clock.toc() << "s\n";
  for (size_t i = 0; i < zz.n_elem; ++i)
    if (std::abs(z[i] - zz[i]) > 1e-5)
      std::cout << "element " << i << " differs: " << z[i] << " vs. " << zz[i]
          << "!" << std::endl;



  return EXIT_SUCCESS;
}
