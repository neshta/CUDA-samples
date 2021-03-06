%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <cstdlib>

#define N           3
#define BLOCK_SIZE	3
#define BASE_TYPE   double

__device__ BASE_TYPE smult(BASE_TYPE *a, BASE_TYPE *b, int idx_a, int idx_b)
{
    BASE_TYPE result = 0;
    for(int i = 0; i < N; i++)
        result += a[idx_a + i] * b[idx_b + i];
    return result;
}

__global__ void ort(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *proj, int step)
{
    int idx_a = step * N;
    int idx_b = blockIdx.x * N;
    for(int i = 0; i < N; i++)
    {
        BASE_TYPE d1, d2;
        d1 = smult(a, b, idx_a, idx_b);
        d2 = smult(b, b, idx_b, idx_b);
        BASE_TYPE d3 = d1 / d2;
        proj[blockIdx.x * N + i] = d3 * b[blockIdx.x * N + i];
        //printf("block %d, threadIdx.x %d: proj[%d] = %.3f, %.2f / %.2f = %.2f\n", blockIdx.x, threadIdx.x, blockIdx.x * N + i, proj[blockIdx.x * N + i], d1, d2, d3);
    }
}

void sub(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *proj, int step)
{
    for(int i = 0; i < step; i++)
    {
        for(int k = 0; k < N; k++)
        {
            if(i > 0)
                b[step * N + k] -= proj[i * N + k];
            else
                b[step * N + k] = a[step * N + k] - proj[i * N + k];
        }
    }
}

void show(BASE_TYPE *a)
{
	for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            printf("%.3f\t", a[i * N + j]);
        printf("\n");
    }
}

void multc(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *c)
{
    *c = 0;
    for(int i = 0; i < N; i++)
    {
        for(int k = 0; k < N; k++)
        {
            *c += a[i * N + k] * b[i * N + k];
        }
    }
    printf("c = %.1f\n", *c);
}

__global__ void transpose(BASE_TYPE *a, BASE_TYPE *b)
{
	int idx_a, idx_b;
	idx_a = blockIdx.x * N + threadIdx.x;
	idx_b = threadIdx.x * N + blockIdx.x;
	b[idx_b] = a[idx_a];
}

__global__ void mult(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *c)
{
	int idx = blockIdx.x * N + threadIdx.x;
	BASE_TYPE sum = 0;
	for(int i = 0; i < N; i++)
	{
		sum += a[blockIdx.x * N + i] * b[i * N + threadIdx.x];
		//if(blockIdx.x == 0 && threadIdx.x == 0)
		//printf("%.3f = %.3f * %.3f += %.3f\n", sum, a[blockIdx.x * N + i], b[i * N + threadIdx.x], a[blockIdx.x * N + i] * b[i * N + threadIdx.x]);
	}
	c[idx] = sum;
	
}

int main()
{
    BASE_TYPE *a, *b, *proj;
    BASE_TYPE *dev_a, *dev_b, *dev_proj;
    size_t size = sizeof(BASE_TYPE) * N * N;

    a = (BASE_TYPE *)malloc(size);
    b = (BASE_TYPE *)malloc(size);
    proj = (BASE_TYPE *)malloc(size);
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_proj, size);

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(j < i)
            {
                a[i * N + j] = 0;
                b[i * N + j] = 0;
            }
            else
            {
            	a[i * N + j] = 1;
                b[i * N + j] = 1;
            }
        }
    }
    for(int i = 0; i < N; i++)
        b[i] = a[i];
    printf("a:\n");
    show(a);
    printf("\n");
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    for(int i = 1; i < N; i++)
    {
        cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
        dim3 threadsPerBlock = dim3(N, 1);
        dim3 blocksPerGrid = dim3(i);
        ort <<< blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_proj, i);
        cudaMemcpy(proj, dev_proj, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(b, dev_b, size, cudaMemcpyDeviceToHost);
        sub(a, b, proj, i);
    }
    printf("b:\n");
    show(b);
    printf("\n\n");

    BASE_TYPE c;
    //multc(a, b, &c);
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    transpose <<< N, N >>> (dev_b, dev_a);
    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
    
    printf("bT:\n");
    show(a);
    printf("\n");
    for(int i = 0; i < N; i++)
    	for(int k = 0; k < N; k++)
    		proj[i * N + k] = 0;
    cudaMemcpy(dev_proj, proj, size, cudaMemcpyHostToDevice);
    mult <<< N, N >>> (dev_b, dev_a, dev_proj);
    cudaMemcpy(proj, dev_proj, size, cudaMemcpyDeviceToHost);


    printf("\nb * bT:\n");
    show(proj);

    /*
    bool hi = true;
    for(int i = 0; i < N && hi; i++)
    {
        for(int k = 0; k < N && hi; k++)
        {
            //if(a[i * N + k] != b[i * N + k])
            if(i == k)
            {
                if(a[i * N + k] != 1)
                hi = false;
            }
            else
            {
                if(a[i * N + k] != 0)
                    hi = false;
            }
        }
    }
    printf("\n%s\n", (hi ? "ok" : "not equal"));
    */
    cudaDeviceSynchronize();
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_proj);
    free(a);
    free(b);
    free(proj);
    return 0;
}


