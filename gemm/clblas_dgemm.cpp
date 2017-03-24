/**
 * @file clblas_gemm.cpp
 * @author Ryan Curtin
 *
 * Perform a GEMM call with square matrices of the size given as argv[1].
 */
#include <sys/types.h>
#include <stdio.h>
#include <clBLAS.h>
#include <armadillo>

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "what is the size of the matrix you want to multiply!?" <<
        std::endl;
    return -1;
  }

  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  int ret = 0;

  // Get size of matrix.
  int size = atoi(argv[1]);

  // Allocate random matrix.
  arma::wall_clock clock;
  clock.tic();
  arma::mat x(size, size, arma::fill::randu);
  arma::mat y(size, size, arma::fill::randu);
  arma::mat z(size, size); // Results.
  std::cout << "matrix initialization time: " << clock.toc() << "s\n";

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  props[1] = (cl_context_properties) platform;
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);

  /* Setup clBLAS */
  err = clblasSetup();

  /* Prepare OpenCL memory objects and place matrices inside them. */
  clock.tic();
  const size_t bufSize = size * size * sizeof(double);
  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bufSize, NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bufSize, NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bufSize, NULL, &err);

  // Now copy the data to the buffer.
  err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bufSize, x.memptr(), 0,
      NULL, NULL);
  err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bufSize, y.memptr(), 0,
      NULL, NULL);
  std::cout << "buffer copy: " << clock.toc() << "s\n";

  // Multiply the two matrices.
  clock.tic();
  err = clblasDgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans, size, size,
      size, 1.0, bufA, 0, size, bufB, 0, size, 0.0, bufC, 0, size, 1, &queue, 0,
      NULL, &event);
  if (err == clblasInvalidDevice)
    std::cerr << "invalid device!\n";
  else if (err == clblasInvalidValue)
    std::cerr << "invalid value!\n";
  else if (err != clblasSuccess)
    std::cerr << "error " << err << "!\n";

  /* Wait for calculations to be finished. */
  err = clWaitForEvents(1, &event);
  std::cout << "multiply time: " << clock.toc() << "s\n";

  /* Fetch results of calculations from GPU memory. */
  err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bufSize, z.memptr(), 0,
      NULL, NULL);

  /* Release OpenCL memory objects. */
  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);

  /* Finalize work with clBLAS */
  clblasTeardown();

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  // Verify correctness.
  clock.tic();
  arma::mat zz = x * y;
  std::cout << "cpu multiply time: " << clock.toc() << "s\n";
  for (size_t i = 0; i < zz.n_elem; ++i)
    if (std::abs(z[i] - zz[i]) > 1e-5)
      std::cout << "element " << i << " differs: " << z[i] << " vs. " << zz[i]
          << "!" << std::endl;

  return ret;
}
