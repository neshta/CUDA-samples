
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
using namespace std;
int N = 100;

int main()
{
    for(; N < 1001; N += 100)
    {
        string text[N];
        for (int i = 0; i < N; i++) {
            text[i] = "albanchino capuchino capuchino albanchino albanchino capuchino capuchino albanchino albanchino capuchino capuchino albanchino albanchino capuchino capuchino albanchino albanchino capuchino capuchino albanchino";
        }

        string host_text[N];
        string* dev_text;

        cudaMalloc((void**)&dev_text, N * sizeof(string));

        for (int i = 0; i < N; i++)
        {
            host_text[i] = text[i];
        }
        cudaEvent_t start, stop;
        float elapsedTime;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMemcpy(dev_text, host_text, N * sizeof(string), cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("N = %d; Symbols: %d\n", N, sizeof(text));
        printf("CPU to GPU: %.4f ms; ", elapsedTime);
        float x = 1 / elapsedTime;
        printf("Speed: %.3f GB/s\n", (float) (sizeof(text)) * x * 1000 / 1024 / 1024 / 1024 );
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaEvent_t start2, stop2;
        float elapsedTime2;

        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
        cudaEventRecord(start2, 0);

        cudaMemcpy(host_text, dev_text, N * sizeof(int),cudaMemcpyDeviceToHost);
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime(&elapsedTime2, start2, stop2);

        printf("GPU to CPU: %.4f ms; ", elapsedTime2);
        x = 1 / elapsedTime2;
        printf("Speed: %.3f GB/s\n\n", (float) (sizeof(text)) * x * 1000 / 1024 / 1024 / 1024 );
        cudaEventDestroy(start2);
        cudaEventDestroy(stop2);
        cudaFree(dev_text);
    }

    return 0;
}
