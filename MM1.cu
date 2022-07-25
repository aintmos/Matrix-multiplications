#include "matConst.hpp"

using namespace std;

void MM_Kernel(dataType** matrix, dataType** input, dataType** res){
    for(int i = 0; i < rowNumC; ++i){
        for(int j = 0; j < colNumC; ++j){
            res[i][j] = 0;
            for(int k = 0; k < colNumA; ++k){
                res[i][j] += matrix[i][k] * input[k][j];
            }
        }
    }
}
