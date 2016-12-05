#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

// This is a kinda naive implementation. At least it works
// Originally want to try pipeline, given up because of data dependency & extra document reading

using namespace std;

int ThreadBlockSize = 256;

__global__ void Reduction_Parallel_Scan(int * data, const int OFFSET) {
    // deal with shared memory
    extern __shared__ int share[];

    share[threadIdx.x] = data[threadIdx.x + OFFSET];
    share[threadIdx.x + blockDim.x] = data[threadIdx.x + blockDim.x + OFFSET];
    __syncthreads();

    int index;
    // This is for upper part of the reduction
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        index = (threadIdx.x + 1)*stride * 2 - 1;
        if (index < 2 * blockDim.x)
            share[index] += share[index - stride];
        __syncthreads();
    }

    // This is for lower part of the reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        index = (threadIdx.x + 1)*stride * 2 - 1;
        if (index + stride < 2 * blockDim.x) {
            share[index + stride] += share[index];
        } 
    }
    __syncthreads();

    data[threadIdx.x + OFFSET] = share[threadIdx.x];
    data[threadIdx.x + blockDim.x + OFFSET] = share[threadIdx.x + blockDim.x];
}

void ScanCUDA(int * A_GPU, int * arr, int N);

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
    A_gpu = new int[N];


    A_cpu[0] = arr[0];
    for (int i = 1; i < N; i++) {
        A_cpu[i] = A_cpu[i - 1] + arr[i];
    }

    // Compute the scan on GPU side
    ScanCUDA(A_gpu, arr, N);

    //compare gpu and cpu result
    for (int i = 0; i < N; i++) {
        if (A_cpu[i] != A_gpu[i]) {

            cerr << "Error @"<<i << endl;
            goto cleanup;
        }
    }
    cout << "Comparision done, no error" << endl;

    cleanup:
    delete[] arr;
    delete[] A_cpu;
    delete[] A_gpu;
}

void ScanCUDA(int * A_GPU, int * arr, int N) {
    int i;

    int DataPerBlock = 2 * ThreadBlockSize;
    int NBlocks = (N - 1) / DataPerBlock + 1;

    int *h_data;
    int *d_data;

    // allocate space
    size_t sharedsize = DataPerBlock * sizeof(int);
    size_t size = NBlocks*sharedsize;
    
    cudaHostAlloc(&h_data, size, cudaHostAllocMapped);

    for (i = 0; i < N; i++) 
        h_data[i] = arr[i];

    cudaHostGetDevicePointer(&d_data, h_data, 0);

    // start calcuation
    for (i = 0; i < NBlocks - 1; i++) {
        Reduction_Parallel_Scan <<<1, ThreadBlockSize, sharedsize>>>(d_data, i*DataPerBlock);
        cudaDeviceSynchronize();
        h_data[(i + 1)* DataPerBlock] += h_data[(i + 1)* DataPerBlock - 1];
    }
    Reduction_Parallel_Scan <<<1, ThreadBlockSize, sharedsize >>>(d_data, i*DataPerBlock);
    cudaDeviceSynchronize();

    // copy back data
    for (i = 0; i < N; i++) 
        A_GPU[i] = h_data[i];

    cudaFreeHost(h_data);
}