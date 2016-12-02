#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// TODO
// thread block size over limit. NEED try pipeline
// deal with smaller than 2^n block size

using namespace std;

int BLOCK_SIZE = 8; // defined blocksize, which is half of the size of the scan array 
// The size of the scan array needs to be power of 2. if not, devides up the scan array to nearest power of 2 and another part
int share[2*10]; // share[2*BLOCK_SIZE] is in shared memory

__global__ void Reduction_Parallel_Scan(int * raw, int * result, const int BLOCK_SIZE) {
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
    
    // deal with shared memory
    extern __shared__ int share[];
    for (unsigned int i = 0; i < 2 * BLOCK_SIZE; i++) {
        share[i] = raw[i];
    }

    int index;
    // This is for upper part of the reduction
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        index = (threadIdx.x + 1)*stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
            share[index] += share[index - stride];
        __syncthreads();
    }

    // This is for lower part of the reduction
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        index = (threadIdx.x + 1)*stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE) {
            share[index + stride] += share[index];
        } 
    }
    __syncthreads();

    result[threadIdx.x] = share[threadIdx.x];
    result[threadIdx.x + BLOCK_SIZE] = share[threadIdx.x + BLOCK_SIZE];
}

void testIndexCorrect() {
    cout << "This is for upper part of the reduction" << endl;
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        for (int i = 0; i < INT_MAX; i++) {
            int index = (i + 1)*stride * 2 - 1;
            if (index < 2 * BLOCK_SIZE)
                cout << index << "\t";
            else break;
        }
        cout << endl;
    }

    cout << "This is for lower part of the reduction" << endl;
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        for (int i = 0; i < INT_MAX; i++) {
            int index = (i + 1)*stride * 2 + stride - 1;
            if (index < 2 * BLOCK_SIZE)
                cout << index << "\t";
            else break;
        }
        cout << endl;
    }
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

    // print check raw array and scan output
    //cout << "Raw array:" << endl;
    //for (int i = 0; i < N; i++) {
    //    cout << arr[i] << "\t";
    //}
    cout << "\nScan array:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << A_cpu[i] << "\t";
    }
    cout << endl;

    //compare gpu and cpu result
    for (int i = 0; i < N; i++) {
        if (A_cpu[i] != A_gpu[i]) {

            cerr << i << endl;
            abort();
        }
    }

    delete[] arr;
    delete[] A_cpu;
    delete[] A_gpu;
}

void ScanCUDA(int * A_GPU, int * arr, int N)
{
    int i;
    int small, large = 1;
    bool exact = false;

    size_t size;
    // some calculation required
    while (large < N) {
        small = large;
        large *= 2;
    }
    if (large == N)
        exact = true;



    // the input size is the same as large or large size is preferd
    if (exact || ((large - N) < (N - small))) {
        cout << "large used" << endl;
        int *h_data, *h_res;
        int *d_data, *d_res;

        size = large * sizeof(int);
        cudaHostAlloc(&h_data, size, cudaHostAllocMapped);
        cudaHostAlloc(&h_res, size, cudaHostAllocMapped);

        for (i = 0; i < N; i++) {
            h_data[i] = arr[i];
        }

        cudaHostGetDevicePointer(&d_data, h_data, 0);
        cudaHostGetDevicePointer(&d_res, h_res, 0);

        Reduction_Parallel_Scan << <1, large / 2, size >> >(d_data, d_res, large / 2);
        cudaDeviceSynchronize();

        for (i = 0; i < 10; i++) {
            cout << h_res[i] << "\t";
        }
        cout << endl;

        for (i = 0; i < N; i++) {
            A_GPU[i] = h_res[i];
        }

        cudaFreeHost(h_data);
    }
    else {
        cout << "small used" << endl;
    }

}
