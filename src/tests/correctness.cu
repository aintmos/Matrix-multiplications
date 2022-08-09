#include <iostream>
#include <time.h>
#include "common.hpp"

void MM_Correct(dataType** matrix, dataType** input, dataType** res,
    const size_t sizeX, const size_t sizeRange, const size_t sizeY){
    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeY; ++j){
            res[i][j] = 0;
            for(int k = 0; k < sizeRange; ++k){
                res[i][j] += matrix[i][k] * input[k][j];
            }
        }
    }
}

using namespace std;
extern float gemm(dataType** matrix, dataType** input, dataType** res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize);

dataType randomF(){
    return (rand()%3) - 1;
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
