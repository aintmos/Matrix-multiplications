#include "GPUdebug.hpp"
#include "matConst.hpp"

using namespace std;

__global__ void MM_Kernel_Kernel(dataType* matrix, dataType* input, dataType* res){
    unsigned int li = threadIdx.x; unsigned int bi = blockIdx.x; unsigned int bis = blockDim.x;
    unsigned int lj = threadIdx.y; unsigned int bj = blockIdx.y; unsigned int bjs = blockDim.y;
    unsigned int globalRowIdx = li + bi * bis;
    unsigned int globalColIdx = lj + bj * bjs;
    int acc = 0;
    for(int k = 0; k < (colNumA + subBlockSize - 1)/subBlockSize; ++k){
        for(int lk = 0; lk < subBlockSize; ++lk){   
            int globalK = subBlockSize * k + lk;
            if(globalK < colNumA)
                acc += matrix[globalRowIdx * colNumA + globalK] * input[globalK * colNumB + globalColIdx];
        }
    }
    if(globalRowIdx < rowNumC && globalColIdx < colNumC)
        res[globalRowIdx * colNumC + globalColIdx] = acc;
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
        HANDLE_ERROR(cudaMemcpy(input_GPU + colNumB * i, input[i], sizeof(dataType)*colNumB, cudaMemcpyHostToDevice));
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
