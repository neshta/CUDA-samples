%%cu
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <cmath>
const int BLOCK_SIZE = 4;

__global__ void Sum_Matrix(int a[][4],int b[][4],int c[][4]) {
	//int i = threadIdx.x;
	//int j = threadIdx.y;
	int i = blockIdx.x;
	int j = blockIdx.y;
	c[i][j] = a[i][j] + b[i][j];
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int n = BLOCK_SIZE;
	int a[BLOCK_SIZE][BLOCK_SIZE] = { {0,2,3,4},
									  {2,6,8,9},
									  {1,2,3,4},
									  {7,2,3,5} };

	int b[BLOCK_SIZE][BLOCK_SIZE] = { {0,4,6,4},
									  {4,6,8,9},
									  {1,2,2,1},
									  {7,2,5,8} };
									
	int c[BLOCK_SIZE][BLOCK_SIZE] = { {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0} };

	int(*pa)[BLOCK_SIZE], (*pb)[BLOCK_SIZE], (*pc)[BLOCK_SIZE];

	cudaMalloc((void**)&pa, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(int));
	cudaMalloc((void**)&pb, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(int));
	cudaMalloc((void**)&pc, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(int));

	cudaMemcpy(pa, a, BLOCK_SIZE * BLOCK_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pb, b, BLOCK_SIZE * BLOCK_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pc, c, BLOCK_SIZE * BLOCK_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(1);

	cudaEventRecord(start, 0);
	Sum_Matrix <<< threadsPerBlock, blocksPerGrid >>> (pa, pb, pc);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(c, pc, BLOCK_SIZE * BLOCK_SIZE* sizeof(int), cudaMemcpyDeviceToHost);
	printf("1: \n");
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf("%d ", a[i][j]);
		}
		printf("\n");
	}

	printf("\n2: \n");
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf("%d ", b[i][j]);
		}
		printf("\n");
	}

	printf("\nresult: \n");
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}

	printf("\nelapsed: %.4f ms\n", elapsed);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return 0;
}
