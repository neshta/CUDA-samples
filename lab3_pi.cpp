
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <clocale>

#define NUM_BLOCK 256 
#define NUM_THREAD 256 
#define NSIZE 268435456 

const double PI = 3.14159265358979323846;

__global__ void calc(double* sum, int size, double step, int threads, int blocks)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double x;
	int i;
	for (i = idx; i < size; i += threads * blocks) 
	{ 
		x = (i + 0.5) * step;
		sum[idx] += 4.0 / (1.0 + x * x);
	}
}

int main()
{
	double pi = 0;
	int tid = 0;
	double *sum, *dev_sum;
	double step = 1.0 / NSIZE;
	size_t size = NUM_BLOCK * NUM_THREAD * sizeof(double);

	sum = (double*)malloc(size);
	cudaMalloc((void**)&dev_sum, size);

	calc <<<NUM_BLOCK, NUM_THREAD>>>(dev_sum, NSIZE, step, NUM_THREAD, NUM_BLOCK);
	cudaMemcpy(sum, dev_sum, size, cudaMemcpyDeviceToHost);

	for (tid = 0; tid < NUM_THREAD * NUM_BLOCK; tid++)
		pi += sum[tid];
	pi *= step;

	printf("%d blocks, %d threads\n", NUM_BLOCK, NUM_THREAD);
	printf("Defined PI    = %.20f\n", PI);
	printf("Calculated PI = %.20f\n", pi);
	printf("Error         = %.20f\n", abs(pi - PI));

	cudaFree(dev_sum);
	return 0;
}
