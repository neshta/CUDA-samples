%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <typeinfo>

#define BASE_TYPE float
#define BLOCKS  1000
#define THREADS 1000
const BASE_TYPE pi = 3.141592653;

__global__ void calc(BASE_TYPE *a, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length)
        a[idx] = __tanf(pi * idx);
}

int main() 
{
    BASE_TYPE *res, *dev_res;
    int length = BLOCKS * THREADS;
    size_t size = length * sizeof(BASE_TYPE);
    res = (BASE_TYPE *)malloc(size);
    cudaMalloc((void**)&dev_res, size);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calc <<< BLOCKS, THREADS >>> (dev_res, length);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(res, dev_res, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_res);

    BASE_TYPE *arr, err = 0;
    arr = (BASE_TYPE *)malloc(size);
    for(int i = 0; i < length; i++)
    {
        arr[i] = abs(tanf(pi * i) - res[i]);
        //printf("%.3f - %.3f = %.3f\n", (float)(i / pi), res[i], arr[i]);
    }
    for(int i = 0; i < length; i++)
        err += arr[i];
    err /= (BASE_TYPE)length;
    printf("err: %.10f\nbase_type: %s\nspent: %.5f ms\n", 
            err, (typeid(BASE_TYPE).name() == typeid(float).name() ? "float" : "double"), elapsedTime);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);
    for(int i = 0; i < length; i++)
        BASE_TYPE x = tan(pi * i);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    
    printf("cpu took %.5f ms", elapsedTime);
}
