
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}


int BLOCK_SIZE = 8; // defined blocksize, which is half of the size of the scan array 
// The size of the scan array needs to be power of 2. if not, devides up the scan array to nearest power of 2 and another part
int XY[2*10]; // XY[2*BLOCK_SIZE] is in shared memory

void Reduction_Parallel_Scan() {
    /////////////////////////////////////////////
    // For example, given a scan array of size 8, the Block size should be 4 (8/2);
    // the reduction is done by split the task into two passes.
    // For the upper part, here is what's going on:
    //      The loop will execute (log2(BLOCK_SIZE) + 1) times. In this example, 3 times.
    //      1st iteration: calculate 1 = 1+0; 3 = 3+2; 5 = 5+4; 7 = 7+6;
    //      2nd iteration: calculate 3 = 3+1 = 3+2+1+0; 7 = 7+5 = 7+6+5+4
    //      3rd iteration: calculate 7 = 7+3 = 7+5 + 3+1 = 7+6+5+4 + 3+2+1+0
    // After the upper part, Index 1,3,7 have the right answer. First pass done.
    // For the lower part, here is what's going on:
    //      The loop will execute (log2(BLOCK_SIZE)) times. In this example, 2 times.
    //      1st iteration: calculate 5 = 5+3 = 5+4 + 3+1 = 5+4 + 3+2 + 1+0
    //      2nd iteration: calculate 2 = 2+1 = 2+1+0; 4 = 4+3 = 4+3+2+1+0; 6 = 6+5 = 6+5+4+3+2+1+0
    // Now all the index has the correct answer, reduction done.
    ///////////////////////////////////////////////
    // __syncthreads() requires between each iterations, as some thread may go way faster than others causing correctness issue.
    ///////////////////////////////////////////////

    // This is for upper part of the reduction
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (threadIdx.x + 1)*stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
            XY[index] += XY[index - stride];
        //__syncthreads();
    }

    // This is for lower part of the reduction
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1)*stride * 2 + stride - 1;
        if (index < 2 * BLOCK_SIZE)
            XY[index] += XY[index - stride];
        //__syncthreads();
    }
}

int main(int argc, char ** argv) {
    int N;  // length of the array
    int * arr; // raw random array
    int * A_cpu; // scan array computed on CPU side
    int * A_gpu; // scan array computed on GPU side

    if (argc != 2) {
        cerr << "No given parameter" << endl;
        exit(1);
    }

    N = atoi(argv[1]);

    // generate the random array of given length, and populate it 
    srand(time(NULL));

    arr = new int[N];
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000 + 1;
    }

    // Compute the scan on CPU side
    A_cpu = new int[N];

    A_cpu[0] = arr[0];
    for (int i = 1; i < N; i++) {
        A_cpu[i] = A_cpu[i - 1] + arr[i];
    }

    // print check raw array and scan output
    cout << "Raw array:" << endl;
    for (int i = 0; i < N; i++) {
        cout << arr[i] << "\t";
    }
    cout << "\nScan array:" << endl;
    for (int i = 0; i < N; i++) {
        cout << A_cpu[i] << "\t";
    }
    cout << endl;


    delete[] arr;
    delete[] A_cpu;
}


//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
