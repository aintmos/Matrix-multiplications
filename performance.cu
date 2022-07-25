#include <iostream>
#include <time.h>
#include "matConst.hpp"
#include "GPUdebug.hpp"

using namespace std;
extern void MM_Kernel(dataType** matrix, dataType** input, dataType** res);

float randomF(){
    return (rand()%100000)/100.0;
}

int main(int argc, char **argv){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    srand(time(NULL));

    dataType **matrix = new dataType*[rowNumA];
    for(int i = 0; i < rowNumA; ++i){
        matrix[i] = new int[colNumA];
        for(int j = 0; j < colNumA; ++j){
            matrix[i][j] = randomF();
        }
    }
    
    dataType **input = new dataType*[rowNumB];
    for(int i = 0; i < rowNumB; ++i){
        input[i] = new int[colNumB];
        for(int j = 0; j < colNumB; ++j){
            input[i][j] = randomF();
        }
    }
    
    dataType **res = new dataType*[rowNumC];
    for(int i = 0; i < rowNumC; ++i){
        res[i] = new int[colNumC];
    }

    HANDLE_ERROR(cudaEventRecord(start));
    MM_Kernel(matrix, input, res);
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << milliseconds << "\n";

    for(int i = 0; i < rowNumA; ++i)
        delete[] matrix[i];
    for(int i = 0; i < rowNumB; ++i)
        delete[] input[i];
    for(int i = 0; i < rowNumC; ++i){
        delete[] res[i];
    }
    delete[] matrix;
    delete[] input;
    delete[] res;
    
    return 0;
}
