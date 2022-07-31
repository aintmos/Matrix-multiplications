#include <iostream>
#include <time.h>
#include "common.hpp"
#include "GPUdebug.hpp"
using namespace std;

__global__ void MM_Kernel__correct(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY){
    size_t global = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = global/sizeY;
    size_t j = global%sizeY;
    global = i * sizeY + j;
    if(global > sizeX * sizeY) return;
    res[global] = 0;
    for(int k = 0; k < sizeRange; ++k){
        res[global] += matrix[i * sizeRange + k] * input[k * sizeY + j];
    }
}

float MM_Correct(dataType** matrix, dataType** input, dataType** res,
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
    HANDLE_ERROR(cudaMalloc(&input_GPU, sizeof(dataType)*rowNumInp*colNumInp));
    HANDLE_ERROR(cudaMalloc(&res_GPU, sizeof(dataType)*rowNumRes*colNumRes));
    for(int i = 0; i < rowNumMat; ++i){
        HANDLE_ERROR(cudaMemcpy(matrix_GPU + colNumMat * i, matrix[i], sizeof(dataType) * colNumMat, cudaMemcpyHostToDevice));
    }
    for(int i = 0; i < rowNumInp; ++i){
        HANDLE_ERROR(cudaMemcpy(input_GPU + colNumInp * i, input[i], sizeof(dataType) * colNumInp, cudaMemcpyHostToDevice));
    }

    int work = rowNumRes * colNumRes;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    HANDLE_ERROR(cudaEventRecord(start));
    MM_Kernel__correct<<<(work + maxThreadSize - 1)/maxThreadSize, work > maxThreadSize? maxThreadSize : work >>>(matrix_GPU, input_GPU, res_GPU, rowSize, rangeSize, colSize);
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);

    for(int i = 0; i < rowNumRes; ++i){
        HANDLE_ERROR(cudaMemcpy(res[i], res_GPU + colNumRes * i, sizeof(dataType) * colNumRes, cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}

extern float gemm(dataType** matrix, dataType** input, dataType** res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize);

float randomF(){
    return (rand()%100000)/100.0;
}

int main(int argc, char **argv){
    srand(time(NULL));

    if(argc != 4)
        return 0;

    size_t sizeX, sizeRange, sizeY;
    sizeX = 0;
    sizeY = 0;
    sizeRange = 0;
    
    char* v = argv[1];
    while(*v != '\0'){
        sizeX *= 10;
        sizeX += *v - '0';
        ++v;
    }
    
    v = argv[2];
    while(*v != '\0'){
        sizeRange *= 10;
        sizeRange += *v - '0';
        ++v;
    }

    v = argv[3];
    while(*v != '\0'){
        sizeY *= 10;
        sizeY += *v - '0';
        ++v;
    }

    dataType **matrix = new dataType*[sizeX];
    for(int i = 0; i < sizeX; ++i){
        matrix[i] = new int[sizeRange];
        for(int j = 0; j < sizeRange; ++j){
            matrix[i][j] = randomF();
        }
    }
    
    dataType **input = new dataType*[sizeRange];
    for(int i = 0; i < sizeRange; ++i){
        input[i] = new int[sizeY];
        for(int j = 0; j < sizeY; ++j){
            input[i][j] = randomF();
        }
    }
    
    dataType **res1 = new dataType*[sizeX];
    for(int i = 0; i < sizeX; ++i){
        res1[i] = new int[sizeY];
    }
    
    dataType **res2 = new dataType*[sizeX];
    for(int i = 0; i < sizeX; ++i){
        res2[i] = new int[sizeY];
    }

    MM_Correct(matrix, input, res1, sizeX, sizeRange, sizeY);
    gemm(matrix, input, res2, sizeX, sizeRange, sizeY);
    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeY; ++j){
            dataType delta = abs(res1[i][j] - res2[i][j]);
            if(delta != 0){
                cout << "Error\n" << i << "," << j << "\n" <<res1[i][j] << " - " << res2[i][j] << " = " << delta << "\n";
                return 0;
            }
        }
    }

    for(int i = 0; i < sizeX; ++i)
        delete[] matrix[i];
    for(int i = 0; i < sizeRange; ++i)
        delete[] input[i];
    for(int i = 0; i < sizeX; ++i){
        delete[] res1[i];
        delete[] res2[i];
    }
    
    delete[] matrix;
    delete[] input;
    delete[] res1;
    delete[] res2;
    
    cout << "Pass\n";
    return 0;
}
