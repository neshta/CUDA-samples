%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <cstdlib>

#define N           1100
#define BASE_TYPE   int

__global__ void mult(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *c)
{
	//int idx = blockIdx.x * N + threadIdx.x;
    int idx = blockIdx.y * N + blockIdx.x;
	BASE_TYPE sum = 0;
	for(int i = 0; i < N; i++)
		sum += a[blockIdx.y * N + i] * b[i * N + blockIdx.x];
	c[idx] = sum;
	//printf("c[%d] += %d * %d += %d = %d\n", idx, a[blockIdx.x * N + i], b[k * N + threadIdx.x], a[blockIdx.x * N + i] * b[k * N + threadIdx.x], sum);
}

void show(BASE_TYPE *a)
{
	for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            printf("%d\t", a[i * N + j]);
        printf("\n");
    }
}

int main()
{
    BASE_TYPE *a, *b, *m1, *m2;
    BASE_TYPE *dev_a, *dev_b, *dev_m1, *dev_m2;
    size_t size = sizeof(BASE_TYPE) * N * N;

    a = (BASE_TYPE *)malloc(size);
    b = (BASE_TYPE *)malloc(size);
    m1 = (BASE_TYPE *)malloc(size);
    m2 = (BASE_TYPE *)malloc(size);
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_m1, size);
    cudaMalloc((void**)&dev_m2, size);

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
        	a[i * N + j] = i * N + j;
            b[i * N + j] = i * N + j;
        }
    printf("1:\n");
    show(a);
    printf("\n2:\n");
    show(a);
    printf("\n");
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    dim3 bl = dim3(N, N);
    dim3 th = dim3(1);
    mult <<< bl, th >>> (dev_a, dev_b, dev_m1);
    cudaMemcpy(m1, dev_m1, size, cudaMemcpyDeviceToHost);
    mult <<< bl, th >>> (dev_b, dev_a, dev_m2);
    cudaMemcpy(m2, dev_m2, size, cudaMemcpyDeviceToHost);
    printf("\nA * B:\n");
    show(m1);
    printf("\nB * A:\n");
    show(m2);
    bool hi = true;
    for(int i = 0; i < N && hi; i++)
    {
        for(int k = 0; k < N && hi; k++)
        {
            if(a[i * N + k] != b[i * N + k])
                hi = false;
        }
    }
    printf("\n%s\n", (hi ? "ok" : "not equal"));
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    free(a);
    free(b);
    free(m1);
    free(m2);
    return 0;
}
