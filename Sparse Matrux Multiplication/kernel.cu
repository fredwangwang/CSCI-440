#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <iostream>
#include <fstream>

using namespace std;

int row, col;

__global__ void TransRowBased(int * raw, int * trans, int row);

__global__ void TransColBased(int * raw, int * trans, int col);


void TransposeCUDA(const int * raw);

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Number of parameters does not match" << endl;
        exit(1);
    }

    ifstream in(argv[1]);
    if (!in) {
        cerr << "Cannot open input matrix" << endl;
        exit(1);
    }

    in >> col;
    in >> row;


    // allocate space to hold raw matrix
    int *raw = new int[row*col];

    // initialize the raw matrix
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            in >> raw[i*col+j];
        }
    }

    in.close();

    // call the function to finish the cuda job
    TransposeCUDA(raw);

    // free spaces
    delete[] raw;
}

__global__ void TransRowBased(int * raw, int * trans, int row)
{
    for (int i = 0; i < row; i++) {
        int id = i * blockDim.x + threadIdx.x;
        trans[threadIdx.x * row + i] = raw[id];
    }
}

__global__ void TransColBased(int * raw, int * trans, int col)
{
    // blockdim.x = rowdim
    for (int i = 0; i < col; i++) {
        int id = i + threadIdx.x * col;
        trans[blockDim.x * i + threadIdx.x] = raw[id];
    }
}

void TransposeCUDA(const int * raw)
{
    int n = row * col;
    size_t size = n * sizeof(int);
    // allocate mem space on device
    int * d_raw;
    cudaMalloc(&d_raw, size);
    int * d_tran;
    cudaMalloc(&d_tran, size);

    // copy raw matrix to device
    cudaMemcpy(d_raw, raw, size, cudaMemcpyHostToDevice);
    
    // invoke kernel
    if (col > row) {
        int threadPerBlock = col;
        TransRowBased << <1, threadPerBlock >> >(d_raw, d_tran, row);
    }
    else {
        int threadPerBlock = row;
        TransColBased << <1, threadPerBlock >> >(d_raw, d_tran, col);
    }

    // copy result to host
    int * h_tran = new int[n];
    cudaMemcpy(h_tran, d_tran, size, cudaMemcpyDeviceToHost);

    cudaFree(d_raw);
    cudaFree(d_tran);

    // write back to file
    ofstream ofile("output.txt");
    if (!ofile) {
        cerr << "Error opening output file" << endl;
        exit(1);
    }

    ofile << row << " " << col << endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            ofile << h_tran[i* row + j] << " ";
        }
        ofile << endl;
    }

    ofile.close();

    cout << "file wrote to \"output.txt\" under the same folder as the program" << endl;
}
