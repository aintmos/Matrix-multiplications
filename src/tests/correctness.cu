#include <iostream>
#include <time.h>
#include "matConst.hpp"

void MM_Correct(dataType** matrix, dataType** input, dataType** res){
    for(int i = 0; i < rowNumC; ++i){
        for(int j = 0; j < colNumC; ++j){
            res[i][j] = 0;
            for(int k = 0; k < colNumA; ++k){
                res[i][j] += matrix[i][k] * input[k][j];
            }
        }
    }
}

using namespace std;
extern void MM_Kernel(dataType** matrix, dataType** input, dataType** res);

float randomF(){
    return (rand()%100000)/100.0;
}

int main(int argc, char **argv){
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
    
    dataType **res1 = new dataType*[rowNumC];
    for(int i = 0; i < rowNumC; ++i){
        res1[i] = new int[colNumC];
    }
    
    dataType **res2 = new dataType*[rowNumC];
    for(int i = 0; i < rowNumC; ++i){
        res2[i] = new int[colNumC];
    }

    MM_Correct(matrix, input, res2);
    MM_Kernel(matrix, input, res1);

    for(int i = 0; i < rowNumC; ++i){
        for(int j = 0; j < colNumC; ++j){
            int delta = abs(res1[i][j] - res2[i][j]);
            if(delta != 0){
                cout << "Error\n" << res1[i][j] << "\n" << res2[i][j] << "\n" << delta << "\n";
                return 0;
            }
        }
    }

    for(int i = 0; i < rowNumA; ++i)
        delete[] matrix[i];
    for(int i = 0; i < rowNumB; ++i)
        delete[] input[i];
    for(int i = 0; i < rowNumC; ++i){
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
