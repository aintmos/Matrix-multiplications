#include <iostream>
#include <time.h>
#include <set>
#include "GPUdebug.hpp"
#include "papi.h"

using namespace std;


//A(Sparse) B = C
constexpr int rowNumA = 2048;
constexpr int colNumA = 1024;
constexpr int rowNumB = colNumA;
constexpr int colNumB = 2048;
constexpr int rowNumC = rowNumA;
constexpr int colNumC = colNumB;
constexpr int maxEdge = 100;

void SPMM_CPU(int* rowPtr, int* colIdx, int* value, int** input, int** res){
    for(int i = 0; i < rowNumA; ++i){
        for(int j = rowPtr[i]; j < rowPtr[i + 1]; ++j){
            for(int k = 0; k < colNumB; ++k){
                res[i][k] += value[j] * input[colIdx[j]][k];
            }
        }
    }
}

__global__ void SPMM_Kernel_kernel(int* rowPtr, int* colIdx, int* value, int** input, int** res){
    for(int i = 0; i < rowNumA; ++i){
        for(int j = rowPtr[i]; j < rowPtr[i + 1]; ++j){
            for(int k = 0; k < colNumB; ++k){
                res[i][k] += value[j] * input[colIdx[j]][k];
            }
        }
    }
}

void SPMM_Kernel(int* rowPtr, int* colIdx, int* value, int** input, int** res){
    int* rowPtr_GPU;
    int* colIdx_GPU;
    int* value_GPU;
    int* input_GPU;
    int* res_GPU;
    
    int numNZ = rowPtr[rowNumA];

    HANDLE_ERROR(cudaMalloc(&rowPtr_GPU, sizeof(dataType) * (rowNumA + 1)));
    HANDLE_ERROR(cudaMalloc(&colIdx_GPU, sizeof(dataType) * numNZ));
    HANDLE_ERROR(cudaMalloc(&value_GPU, sizeof(dataType) * numNZ));
    HANDLE_ERROR(cudaMalloc(&input_GPU, sizeof(dataType) * rowNumA * rowNumA));
    HANDLE_ERROR(cudaMalloc(&res_GPU, sizeof(dataType) * rowNumA * rowNumA));

    HANDLE_ERROR(cudaMemcpy(rowPtr_GPU, rowPtr, sizeof(dataType) * (rowNumA + 1), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(colIdx_GPU, colIdx, sizeof(dataType) * numNZ, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(value_GPU, value, sizeof(dataType) * numNZ, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(input_GPU, input, sizeof(dataType) * rowNumA * rowNumA, cudaMemcpyDeviceToHost));

    
    
    HANDLE_ERROR(cudaMemcpy(res, res_GPU, sizeof(dataType) * rowNumA * rowNumA, cudaMemcpyHostToDevice));
    
    HANDLE_ERROR(cudaFree(rowPtr_GPU));
    HANDLE_ERROR(cudaFree(colIdx_GPU));
    HANDLE_ERROR(cudaFree(value_GPU));
      HANDLE_ERROR(cudaFree(input_GPU));
   HANDLE_ERROR(cudaFree(res_GPU));
}


int randomF(){
    return (rand()%100)/10.0f;
}

int main(int argc, char **argv){
    srand(time(NULL));
    int pre = 0;
    int* rowPtr = new int[rowNumA + 1];
    {
        for(int i = 0; i < rowNumA; ++i){
            int range = rand()%(maxEdge - 1) + 1;
            rowPtr[i] = pre;
            pre += range;
        }
        rowPtr[rowNumA] = pre;
    }
    int dataNum = pre;
    int* colIdx = new int[dataNum];
    int* value = new int[dataNum];
    
    for(int i = 0; i < rowNumA; ++i){
        int colNum = rowPtr[i + 1] - rowPtr[i];
        std::set<int> colIdxSet;
        for(int j = 0; j < colNum; ++j){
            int newColIdx;
            do{
                newColIdx = rand() % colNumA;
            }while(colIdxSet.find(newColIdx) != colIdxSet.end());
            colIdxSet.insert(newColIdx);
        }
        for(auto iter = std::make_pair(rowPtr[i], colIdxSet.begin()); iter.second != colIdxSet.end(); ++iter.first, ++iter.second){
            colIdx[iter.first] = *(iter.second);
            value[iter.first] = randomF();
        }
    }

    
    int **input = new int*[rowNumB];
    for(int i = 0; i < rowNumB; ++i){
        input[i] = new int[colNumB];
        for(int j = 0; j < colNumB; ++j){
            input[i][j] = randomF();
        }
    }
    
    int **res = new int*[rowNumC];
    for(int i = 0; i < rowNumC; ++i){
        res[i] = new int[colNumC];
        for(int j = 0; j < colNumC; ++j){
            res[i][j] = 0;
        }
    }


    SPMM_CPU(rowPtr, colIdx, value, input, res);

    for(int i = 0; i < rowNumB; ++i){
        delete[] input[i];
    }

    delete[] input;

    delete[] rowPtr;
    delete[] colIdx;
    delete[] value;

    for(int i = 0; i < rowNumC; ++i){
        delete[] res[i];
    }
    delete[] res;
    
    return 0;
}
