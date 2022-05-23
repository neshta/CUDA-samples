#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <clocale>
#include <cmath>

#define NUM_BLOCK 128 
#define NUM_THREAD 128

const double PI = 3.14159265358979323846;

__global__ void calc(double *res, double step, int blocks, int threads, int size)
{
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx <= size;
		idx += blocks * threads)
	{
		double x = step * idx;
		res[idx] += sqrt(1 - x * x);
	}
}

int main()
{
	const int N = NUM_BLOCK * NUM_THREAD;
	double pi = 0;
	double *res, *dev_res;
	double step = 1.0 / N;
	size_t size = N * sizeof(double);
	res = (double*)malloc(size);
	cudaMalloc((void**)&dev_res, size);

	calc <<<NUM_BLOCK, NUM_THREAD>>> (dev_res, step, NUM_BLOCK, NUM_THREAD, N);
	cudaMemcpy(res, dev_res, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
		pi += res[i];
	pi *= 4 * step;
	pi /= N;

	printf("%d blocks, %d threads\n", NUM_BLOCK, NUM_THREAD);
	printf("Defined PI    = %.10f\n", PI);
	printf("Calculated PI = %.10f\n", pi);
	printf("Error         = %.10f\n", abs(pi - PI));

	cudaFree(dev_res);
	return 0;
}

