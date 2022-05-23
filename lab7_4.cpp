%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N 5
__global__ void add(int* a, int* b, int *c)
{
	__shared__ int ash[N];
	__shared__ int bsh[N];
	ash[threadIdx.x] = a[threadIdx.x];
	bsh[threadIdx.x] = b[threadIdx.x];
	atomicAdd(c, ash[threadIdx.x] * bsh[threadIdx.x]);
}

int main()
{
	int *a, c = 12345;
	int *d_a, *d_c;
	int size = sizeof(int);
	a = (int*)malloc(size * N);
	cudaMalloc((void**)&d_a, size * N);
	cudaMalloc((void**)&d_c, size);
	for(int i = 0; i < N; i++)
		a[i] = i;
	cudaMemcpy(d_a, a, size * N, cudaMemcpyHostToDevice);
	add <<<1, N>>>(d_a, d_a, d_c);
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_c); 
	double res = sqrt(c);
	printf("sqrt(%d) = %.5f", c, res);
	return 0;
}