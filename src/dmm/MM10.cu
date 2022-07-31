#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;

__global__ void MM_Kernel(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY){
    size_t global = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = global/sizeY;
    size_t j = global%sizeY;
    global = i * sizeY + j;
    if(global > sizeX * sizeY) return;
    dataType acc = 0;
    for(int k = 0; k < sizeRange; ++k){
        acc +=  matrix[i * sizeRange + k] * input[k * sizeY + j];
    }
    res[global] = acc;
}
__global__ void AM_Kernel(dataType *from, size_t fromUnit,
            dataType *to,   size_t toUnit,
            size_t height, size_t width){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= height || j >= width) return;
    to[i * toUnit + j] += from[i * fromUnit + j];
}

__global__ void SM_Kernel(dataType *from, size_t fromUnit,
            dataType *to,   size_t toUnit,
            size_t height, size_t width){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= height || j >= width) return;
    to[i * toUnit + j] -= from[i * fromUnit + j];
}

bool isBaseCase(size_t sizeX, size_t sizeRange, size_t sizeY){
    return sizeX * sizeRange<= 1048576 || sizeRange * sizeY <= 1048576 || sizeX * sizeY <= 1048576 ||
        sizeX <= 32 || sizeRange <= 32 || sizeY <= 32;
    //return sizeX <= 1024 || sizeRange <= 1024 || sizeY <= 1024;
}

//Copy array from "from" to "to"
//Index will starts from "fromOffset" and "toOffset"
//It will copy width in a row for height times with each corresponding units.
void addMat(dataType *from, size_t fromUnit,
            dataType *to, size_t toUnit,
            size_t height, size_t width, bool plus){
    dim3 threadDim(height > subBlockSize ? subBlockSize : height , width > subBlockSize ? subBlockSize : width);
    dim3 blockDim((height + subBlockSize - 1)/subBlockSize , (width + subBlockSize - 1)/subBlockSize);
    if(plus)
        AM_Kernel<<<blockDim, threadDim>>>(from, fromUnit, to, toUnit, height, width);
    else
        SM_Kernel<<<blockDim, threadDim>>>(from, fromUnit, to, toUnit, height, width);
}

void clearMat(dataType *mat, size_t matSize){
    HANDLE_ERROR(cudaMemset(mat, 0,  sizeof(dataType) * matSize));
}

void gemm(  dataType * matrix, dataType * input, dataType * result,
            size_t sizeX, size_t sizeRange, size_t sizeY ){
    bool baseCase = isBaseCase(sizeX, sizeRange, sizeY);
    if(baseCase){
        size_t work = sizeX * sizeY;
        MM_Kernel<<<(work + maxThreadSize - 1)/maxThreadSize, work > maxThreadSize? maxThreadSize : work >>>(matrix, input, result, sizeX, sizeRange, sizeY);
    }
    else{
        const size_t rowNumMat = sizeX;
        const size_t rowNumInp = sizeRange;
        const size_t rowNumRes = sizeX;
        const size_t colNumMat = sizeRange;
        const size_t colNumInp = sizeY;
        const size_t colNumRes = sizeY;

        const size_t matSize = colNumMat * rowNumMat;
        const size_t inpSize = colNumInp * rowNumInp;
        const size_t resSize = colNumRes * rowNumRes;
        
        const size_t rowNumAuxmat = (rowNumMat + 1)/2;
        const size_t rowNumAuxinp = (rowNumInp + 1)/2;
        const size_t rowNumAuxres = (rowNumRes + 1)/2;
        const size_t colNumAuxmat = (colNumMat + 1)/2;
        const size_t colNumAuxinp = (colNumInp + 1)/2;
        const size_t colNumAuxres = (colNumRes + 1)/2;

        const size_t rowNumAuxmatLeft = rowNumMat - rowNumAuxmat;
        const size_t rowNumAuxinpLeft = rowNumInp - rowNumAuxinp;
        const size_t rowNumAuxresLeft = rowNumRes - rowNumAuxres;
        const size_t colNumAuxmatLeft = colNumMat - colNumAuxmat;
        const size_t colNumAuxinpLeft = colNumInp - colNumAuxinp;
        const size_t colNumAuxresLeft = colNumRes - colNumAuxres;
        
        const size_t auxMatSize = rowNumAuxmat * colNumAuxmat;
        const size_t auxInpSize = rowNumAuxinp * colNumAuxinp;
        const size_t auxResSize = rowNumAuxres * colNumAuxres;

        const size_t auxSizeX     = rowNumAuxres;
        const size_t auxSizeY     = colNumAuxres;
        const size_t auxSizeRange = colNumAuxmat;
        
        const size_t auxSizeXLeft     = rowNumAuxresLeft;
        const size_t auxSizeYLeft     = colNumAuxresLeft;
        const size_t auxSizeRangeLeft = rowNumAuxmatLeft;

        dataType* auxMat = matrix + matSize;
        dataType* auxInp = input  + inpSize;
        dataType* auxRes = result + resSize;

        dataType* mat_11 = matrix;
        dataType* mat_12 = mat_11 + colNumAuxmat;
        dataType* mat_21 = mat_11 + colNumMat * rowNumAuxmat;
        dataType* mat_22 = mat_21 + colNumAuxmat;

        dataType* inp_11 = input;
        dataType* inp_12 = inp_11 + colNumAuxinp;
        dataType* inp_21 = inp_11 + colNumInp * rowNumAuxinp;
        dataType* inp_22 = inp_21 + colNumAuxinp;

        dataType* res_11 = result;
        dataType* res_12 = res_11 + colNumAuxres;
        dataType* res_21 = res_11 + colNumRes * rowNumAuxres;
        dataType* res_22 = res_21 + colNumAuxres;
        
        ////////////////////////////////////////////////
        // M1 applied
        // (mat_11 + mat_22)(inp_11 + inp_22)
        // positive: res_11, res_22
        // negative: none
        ////////////////////////////////////////////////
        clearMat(auxMat, auxMatSize);
        clearMat(auxInp, auxInpSize);
        clearMat(auxRes, auxResSize);

        addMat( mat_11, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmat, colNumAuxmat, true);

        addMat( mat_22, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmatLeft, colNumAuxmatLeft, true);

        addMat( inp_11,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinp, colNumAuxinp, true);

        addMat( inp_22,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinpLeft, colNumAuxinpLeft,true);
        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);
        
        addMat( auxRes,  colNumAuxres, 
                res_11, colNumRes,
                rowNumAuxres, colNumAuxres, true);

        addMat( auxRes,  colNumAuxres, 
                res_22, colNumRes,
                rowNumAuxresLeft, colNumAuxresLeft, true);
        ////////////////////////////////////////////////     
        // M3 applied
        // (mat_11)(inp_12 - inp_22)
        // positive: res_12, res_22
        // negative: none
        ////////////////////////////////////////////////

        clearMat(auxMat, auxMatSize);
        clearMat(auxInp, auxInpSize);
        clearMat(auxRes, auxResSize);

        addMat( mat_11, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmat, colNumAuxmat, true);

        addMat( inp_12,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinp, colNumAuxinpLeft, true);

        addMat( inp_22,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinpLeft, colNumAuxinpLeft, false);

        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);


        addMat( auxRes,  colNumAuxres, 
                res_12, colNumRes,
                rowNumAuxres, colNumAuxresLeft, true);

        addMat( auxRes,  colNumAuxres, 
                res_22, colNumRes,
                rowNumAuxresLeft, colNumAuxresLeft, true);

        ////////////////////////////////////////////////     
        // M5 applied
        // (mat_11 + mat_12)(inp_22)
        // positive: res_12
        // negative: res_11
        ////////////////////////////////////////////////

        clearMat(auxInp, auxInpSize);
        clearMat(auxRes, auxResSize);

        addMat( mat_12, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmat, colNumAuxmatLeft, true);

        addMat( inp_22,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinpLeft, colNumAuxinpLeft, true);

        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);

        addMat( auxRes,  colNumAuxres, 
                res_12, colNumRes,
                rowNumAuxres, colNumAuxresLeft, true);

        addMat( auxRes,  colNumAuxres, 
                res_11, colNumRes,
                rowNumAuxres, colNumAuxres, false);

        ////////////////////////////////////////////////     
        // M7 applied
        // (mat_12 - mat_22)(inp_21 + inp_22)
        // positive: res_11
        // negative: none
        ////////////////////////////////////////////////
        clearMat(auxMat, auxMatSize);
        clearMat(auxRes, auxResSize);

        addMat( mat_12, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmat, colNumAuxmatLeft, true);

        addMat( mat_22, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmatLeft, colNumAuxmatLeft, false);

        addMat( inp_21,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinpLeft, colNumAuxinp, true);

        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);

        addMat( auxRes,  colNumAuxres, 
                res_11, colNumRes,
                rowNumAuxres, colNumAuxres, true);
        ////////////////////////////////////////////////     
        // M4 applied
        // (mat_22)(- inp_11 + inp_21)
        // positive: res_11, res_21
        // negative: none
        ////////////////////////////////////////////////
        clearMat(auxMat, auxMatSize);
        clearMat(auxInp, auxInpSize);
        clearMat(auxRes, auxResSize);
        
        addMat( mat_22, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmatLeft, colNumAuxmatLeft, true);

        addMat( inp_11,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinp, colNumAuxinp, false);

        addMat( inp_21,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinpLeft, colNumAuxinp, true);

        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);

        addMat( auxRes,  colNumAuxres, 
                res_11, colNumRes,
                rowNumAuxres, colNumAuxres, true);

        addMat( auxRes,  colNumAuxres, 
                res_21, colNumRes,
                rowNumAuxresLeft, colNumAuxres, true);
                
        ////////////////////////////////////////////////     
        // M2 applied
        // (mat_21 + mat_22)(inp_11)
        // positive: res_21
        // negative: res_22
        ////////////////////////////////////////////////
        clearMat(auxInp, auxInpSize);
        clearMat(auxRes, auxResSize);

        addMat( mat_21, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmatLeft, colNumAuxmat, true);

        addMat( inp_11,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinp, colNumAuxinp, true);

        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);

        addMat( auxRes,  colNumAuxres, 
                res_21, colNumRes,
                rowNumAuxresLeft, colNumAuxres, true);

        addMat( auxRes,  colNumAuxres, 
                res_22, colNumRes,
                rowNumAuxresLeft, colNumAuxresLeft, false);

        ////////////////////////////////////////////////     
        // M6 applied
        // (-mat_11 + mat_21)(inp_11 + inp_12)
        // positive: res_22
        // negative: none
        ////////////////////////////////////////////////
        clearMat(auxMat, auxMatSize);
        clearMat(auxRes, auxResSize);

        addMat( mat_11, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmat, colNumAuxmat, false);

        addMat( mat_21, colNumMat, 
                auxMat, colNumAuxmat,
                rowNumAuxmatLeft, colNumAuxmat, true);

        addMat( inp_12,  colNumInp, 
                auxInp, colNumAuxinp,
                rowNumAuxinp, colNumAuxinpLeft, true);

        gemm( auxMat, auxInp, auxRes, auxSizeX, auxSizeRange, auxSizeY);

        addMat( auxRes,  colNumAuxres, 
                res_22, colNumRes,
                rowNumAuxresLeft, colNumAuxresLeft, true);
    }
}


float gemm(dataType** matrix, dataType** input, dataType** res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize){
    dataType *matrix_GPU, *input_GPU, *res_GPU;
    
    const size_t rowNumMat = rowSize;
    const size_t rowNumInp = rangeSize;
    const size_t rowNumRes = rowSize;
    const size_t colNumMat = rangeSize;
    const size_t colNumInp = colSize;
    const size_t colNumRes = colSize;

    //Get auxiliary requirement size
    size_t auxSize_Mat = 0;
    size_t auxSize_Inp = 0;
    size_t auxSize_Res = 0;
    {
        size_t auxRow = rowSize;
        size_t auxRange = rangeSize;
        size_t auxCol = colSize;
        while(!isBaseCase(auxRow, auxRange, auxCol)){//Minimum matrix multiplication is 1024
            auxRow = (auxRow + 1)/2;
            auxCol = (auxCol + 1)/2;
            auxRange = (auxRange + 1)/2;
            
            auxSize_Mat += auxRow * auxRange;
            auxSize_Inp += auxRange * auxCol;
            auxSize_Res += auxRow * auxCol;
        }
    }

    HANDLE_ERROR(cudaMalloc(&matrix_GPU, sizeof(dataType)*(rowNumMat*colNumMat + auxSize_Mat)));
    HANDLE_ERROR(cudaMalloc(&input_GPU,  sizeof(dataType)*(rowNumInp*colNumInp + auxSize_Inp)));
    HANDLE_ERROR(cudaMalloc(&res_GPU,    sizeof(dataType)*(rowNumRes*colNumRes + auxSize_Res)));
    HANDLE_ERROR(cudaMemset(res_GPU, 0,  sizeof(dataType)*(rowNumRes*colNumRes)));

    for(int i = 0; i < rowNumMat; ++i){
        HANDLE_ERROR(cudaMemcpy(matrix_GPU + colNumMat * i, matrix[i], sizeof(dataType)*colNumMat, cudaMemcpyHostToDevice));
    }
    for(int i = 0; i < rowNumInp; ++i){
        HANDLE_ERROR(cudaMemcpy(input_GPU + colNumInp * i, input[i], sizeof(dataType)*colNumInp, cudaMemcpyHostToDevice));
    }

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    HANDLE_ERROR(cudaEventRecord(start));
    gemm(   matrix_GPU, input_GPU, res_GPU,
            rowNumMat, colNumMat, colNumInp);
    
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