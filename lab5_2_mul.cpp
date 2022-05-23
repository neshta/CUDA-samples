%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 100

__global__ void add(int* a, int* b, int *c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//*c += a[idx] * b[idx];
	atomicAdd(c, a[idx] * b[idx]);
	//printf("%d * %d += %d\n", a[idx], b[idx], *c);
}

int main()
{
	int *a, *b, c = 12345;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	a = (int*)malloc(size * N);
	b = (int*)malloc(size * N);

	cudaMalloc((void**)&d_a, size * N);
	cudaMalloc((void**)&d_b, size * N);
	cudaMalloc((void**)&d_c, size);
	
	for(int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i;
	}
	cudaMemcpy(d_a, a, size * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * N, cudaMemcpyHostToDevice);

	add <<<1, N>>>(d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c); 

	printf("%i", c);
	return 0;
}