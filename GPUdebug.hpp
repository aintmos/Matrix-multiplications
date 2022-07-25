#include <iostream>
#ifndef GPUDEBUG
#define GPUDEBUG

static void HandleError(cudaError_t err,const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif