%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <clocale>

#define N 				1000000
#define THREADS 		1000
#define BASE_TYPE int

__global__ void add( BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *c )
{

	int tid = blockIdx.x * THREADS + threadIdx.x;
    if(tid > N-1) return;
	atomicAdd(c, a[tid] * b[tid]);
}

__global__ void scalMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C)
{
	__shared__ BASE_TYPE ash[THREADS];
	__shared__ BASE_TYPE bsh[THREADS];
	int idx;
	idx = blockIdx.x * THREADS + threadIdx.x;
	ash[threadIdx.x] = A[idx];
	bsh[threadIdx.x] = B[idx];
	__syncthreads();
	atomicAdd(C, ash[threadIdx.x] * bsh[threadIdx.x]);
}

int main() {

	BASE_TYPE host_a[N], host_b[N], host_c = 0;
	BASE_TYPE *dev_a, *dev_b, *dev_c;

    for (int i=0; i<N; i++)
	{
	    host_a[i] = i;
		host_b[i] = i;
	}
	cudaMalloc( (void**)&dev_a, N * sizeof(BASE_TYPE) );
	cudaMalloc( (void**)&dev_b, N * sizeof(BASE_TYPE) );
	cudaMalloc( (void**)&dev_c, sizeof(BASE_TYPE) );

	cudaMemcpy( dev_a, host_a, N * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_b, host_b, N * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsed;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int HoJlb = 0;
	for(int i = 0; i < 1000; i++)
	{
		dim3 blocks = dim3(N / THREADS);
		cudaMemcpy(dev_c, &HoJlb, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
		add <<< blocks, THREADS >>> (dev_a, dev_b, dev_c);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Add: %.4f ms for 100 repeats\n", elapsed);
	cudaMemcpy( &host_c, dev_c, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
	printf( "%d\n", host_c);

	cudaEventRecord(start, 0);
	for(int i = 0; i < 1000; i++)
	{
		dim3 blocks = dim3(N / THREADS);
		cudaMemcpy(dev_c, &HoJlb, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
		scalMult <<< blocks, THREADS >>> (dev_a, dev_b, dev_c);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Scal: %.4f ms for 100 repeats\n", elapsed);

	cudaMemcpy( &host_c, dev_c, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
	printf( "%d", host_c);
		
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	return 0;
}