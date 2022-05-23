%%cu
//4
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 1000
#define R 1.0
#define THREAD 50

#define SUPER

const double PI = 3.14159265358979323846;

__device__ double circle(double x, double radius)
{
    double y = radius * radius - x * x; 
    return y; 
}

__device__ double randNum()
{
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    double x = (double)(curand_uniform(&state));
    return x;
}

__global__ void calc(int *count, int size, int* called)
{
    double x, y;
    //x = (double)blockIdx.x / size;
    //y = (double)threadIdx.x / size;

    x = randNum();
    y = randNum();
    if(y * y <= circle(x, R))
    {
        atomicAdd(count, 1);
        atomicAdd(called, 1);
    }
}

__global__ void calc2(int *count, int size, int* called)
{
    double x, y;
    int good = 0;
    for(int i = 0; i < THREAD; i++)
    {
        x = randNum();
        y = randNum();
        if(y * y <= circle(x, R))
        {
            good++;
        }
    }
    atomicAdd(count, good);
    atomicAdd(called, 1);
}

int main() 
{
    int size = N * N;
    double pi = 0;
    int count = 0, *dev_count, called = 0, *dev_called;
    cudaMalloc((void**)&dev_count, sizeof(int));
    cudaMalloc((void**)&dev_called, sizeof(int));
 
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
 
    #if defined (SUPER)
    calc2 <<< N, N / THREAD >>> (dev_count, N, dev_called);
    #else
    calc <<< N, N >>> (dev_count, N, dev_called);
    #endif
   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&called, dev_called, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_count);
    cudaFree(dev_called);
    pi = 4 * (double)count / (double)size;

    printf("%d x %d\nInside: %d / %d\nCalculated PI: %.10f\nDefined PI: %.10f\nError: %.10f\n\nAtomic was called %d times\nElapsed %.5f ms", 
            N, N, count, size, pi, PI, abs(pi - PI), called, elapsedTime);
}
