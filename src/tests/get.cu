#include <fstream>
#include <time.h>
#include "GPUdebug.hpp"
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

int preseed[] = {0,-1,-1,0,0,1,0,1,-1,1,1,-1,1,1,1,1,1,-1,0,1,0,1,-1,1,0,-1,1,0,0,-1,0,-1,0,1,1,1,1,0,1,1,0,0,-1,-1,1,-1,1,1,-1,-1,1,1,0,-1,1,1,0,1,-1,-1,-1,1,0,0,0,0,0,-1,1,0,-1,-1,-1,-1,0,1,0,1,1,1,1,1,1,0,-1,-1,-1,0,1,0,0,0,0,-1,1,1,0,0};
int idx = 0;
float randomF(){
    return 1;
    return preseed[idx++];
    return (rand()%3 - 1);
}
int main(int argc, char **argv){
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    srand(time(NULL));

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
    
    dataType **res = new dataType*[sizeX];
    for(int i = 0; i < sizeX; ++i){
        res[i] = new int[sizeY];
    }

    dataType **ans = new dataType*[sizeX];
    for(int i = 0; i < sizeX; ++i){
        ans[i] = new int[sizeY];
    }

    
    gemm(matrix, input, res, sizeX, sizeRange, sizeY);
    MM_Correct(matrix, input, ans, sizeX, sizeRange, sizeY);

    ofstream writeFile("res.txt");
    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeRange; ++j){
            writeFile << matrix[i][j] << "\t";
        }
        writeFile << "\n";
    }
    writeFile << "\n";
    writeFile << "\n";

    for(int i = 0; i < sizeRange; ++i){
        for(int j = 0; j < sizeY; ++j){
            writeFile << input[i][j] << "\t";
        }
        writeFile << "\n";
    }
    writeFile << "\n";
    writeFile << "\n";

    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeY; ++j){
            writeFile << res[i][j] << "\t";
        }
        writeFile << "\n";
    }
    writeFile << "\n";
    writeFile << "\n";

    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeY; ++j){
            writeFile << ans[i][j] << "\t";
        }
        writeFile << "\n";
    }
    writeFile.close();

    for(int i = 0; i < sizeX; ++i)
        delete[] matrix[i];
    for(int i = 0; i < sizeRange; ++i)
        delete[] input[i];
    for(int i = 0; i < sizeX; ++i){
        delete[] res[i];
    }
    delete[] matrix;
    delete[] input;
    delete[] res;
    
    return 0;
}
