#include "GPUdebug.hpp"
#include "matConst.hpp"

using namespace std;

__global__ void MM_Kernel_Kernel(dataType* matrix, dataType* input, dataType* res){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= rowNumC || j >= colNumC) return;

    int acc = 0;
    for(int k = 0; k < colNumA; ++k){
        acc += matrix[i * colNumA + k] * input[k * colNumB + j];
    }
    res[i * colNumC + j] = acc;
}

void MM_Kernel(dataType** matrix, dataType** input, dataType** res){
    dataType *matrix_GPU;
    dataType *input_GPU;
    dataType *res_GPU;
    HANDLE_ERROR(cudaMalloc(&matrix_GPU, sizeof(dataType)*rowNumA*colNumA));
    HANDLE_ERROR(cudaMalloc(&input_GPU,  sizeof(int)*rowNumB*colNumB));
    HANDLE_ERROR(cudaMalloc(&res_GPU,    sizeof(int)*rowNumC*colNumC));
    for(int i = 0; i < rowNumA; ++i){
        HANDLE_ERROR(cudaMemcpy(matrix_GPU + colNumA * i, matrix[i], sizeof(dataType)*colNumA, cudaMemcpyHostToDevice));
    }
    for(int i = 0; i < rowNumB; ++i){
        HANDLE_ERROR(cudaMemcpy(input_GPU  + colNumB * i, input[i],  sizeof(int)*colNumB, cudaMemcpyHostToDevice));
    }

    dim3 threadDim(rowNumC > subBlockSize ? subBlockSize : rowNumC , colNumC > subBlockSize ? subBlockSize : colNumC);
    dim3 blockDim((rowNumC + subBlockSize - 1)/subBlockSize , (colNumC + subBlockSize - 1)/subBlockSize);
    MM_Kernel_Kernel<<<blockDim, threadDim>>>(matrix_GPU, input_GPU, res_GPU);

    for(int i = 0; i < rowNumC; ++i){
        HANDLE_ERROR(cudaMemcpy(res[i], res_GPU + colNumC * i, sizeof(dataType)*colNumC, cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
}
