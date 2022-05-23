#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define NUM_BLOCK 100
#define NUM_THREAD 10
#define POWER 2

__global__ void add (int k, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	c[i] = powf(i, -k);
}

int main()
{
	int size = NUM_BLOCK * NUM_THREAD;
	float c[size];
	float *dev_c; 
	cudaMalloc((void**)&dev_c, size * sizeof(float));

	add <<< NUM_BLOCK, NUM_THREAD >>> (POWER, dev_c);

	cudaMemcpy(&c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	float res = 0;
	for (int j = 0; j < size; j++)
		res += c[j];
	printf("result is %f\n", res);

	cudaFree(dev_c);
	return 0;
}



