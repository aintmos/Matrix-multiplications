#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;

__global__ void MM_Kernel(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY){
    size_t li = threadIdx.x; size_t bi = blockIdx.x; size_t bis = blockDim.x;
    size_t lj = threadIdx.y; size_t bj = blockIdx.y; size_t bjs = blockDim.y;
    size_t globalRowIdx = li + bi * bis;
    size_t globalColIdx = lj + bj * bjs;
    dataType acc = 0;
    for(int k = 0; k < (sizeRange + subBlockSize - 1)/subBlockSize; ++k){
        for(int lk = 0; lk < subBlockSize; ++lk){   
            size_t globalK = subBlockSize * k + lk;
            if(globalK < sizeRange)
                acc += matrix[globalRowIdx * sizeRange + globalK] * input[globalK * sizeY + globalColIdx];
        }
    }
    if(globalRowIdx < sizeX && globalColIdx < sizeY)
        res[globalRowIdx * sizeY + globalColIdx] = acc;
}

float gemm(dataType** matrix, dataType** input, dataType** res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize){
    const size_t rowNumMat = rowSize;
    const size_t rowNumInp = rangeSize;
    const size_t rowNumRes = rowSize;
    const size_t colNumMat = rangeSize;
    const size_t colNumInp = colSize;
    const size_t colNumRes = colSize;
    dataType *matrix_GPU;
    dataType *input_GPU;
    dataType *res_GPU;
    HANDLE_ERROR(cudaMalloc(&matrix_GPU, sizeof(dataType)*rowNumMat*colNumMat));
    HANDLE_ERROR(cudaMalloc(&input_GPU,  sizeof(int)*rowNumInp*colNumInp));
    HANDLE_ERROR(cudaMalloc(&res_GPU,    sizeof(int)*rowNumRes*colNumRes));
    for(int i = 0; i < rowNumMat; ++i){
        HANDLE_ERROR(cudaMemcpy(matrix_GPU + colNumMat * i, matrix[i], sizeof(dataType)*colNumMat, cudaMemcpyHostToDevice));
    }
    for(int i = 0; i < rowNumInp; ++i){
        HANDLE_ERROR(cudaMemcpy(input_GPU + colNumInp * i, input[i], sizeof(dataType)*colNumInp, cudaMemcpyHostToDevice));
    }

    dim3 threadDim(rowNumRes > subBlockSize ? subBlockSize : rowNumRes , colNumRes > subBlockSize ? subBlockSize : colNumRes);
    dim3 blockDim((rowNumRes + subBlockSize - 1)/subBlockSize , (colNumRes + subBlockSize - 1)/subBlockSize);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    HANDLE_ERROR(cudaEventRecord(start));
    MM_Kernel<<<blockDim, threadDim>>>(matrix_GPU, input_GPU, res_GPU, rowSize, rangeSize, colSize);
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    for(int i = 0; i < rowNumRes; ++i){
        HANDLE_ERROR(cudaMemcpy(res[i], res_GPU + colNumRes * i, sizeof(dataType)*colNumRes, cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}
