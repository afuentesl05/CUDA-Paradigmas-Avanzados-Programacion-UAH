#include <cstdio>
#include <cuda_runtime.h>

__global__ void basicKernel() {
    int blq = blockIdx.x;
    int dim = blockDim.x;
    int tId = threadIdx.x;
    int index = blq * dim + tId;

    printf("Thread #%d. Bloque #%d, hilo #%d\n", index, blq, tId);
}

int main() {
    basicKernel << <3, 4 >> > ();
    cudaDeviceSynchronize();
    return 0;
}

