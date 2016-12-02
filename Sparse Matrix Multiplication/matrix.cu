#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ThreadBlockSize 256

__global__ void SparseMatVecCUDA(int *ptr, int *index, float *data, float *b, float *t, const int n, const int numRow)
{
    extern __shared__ float result[];

    int ind;

    for (int i = 0; i < n; i += ThreadBlockSize) {
        ind = i + threadIdx.x;
        if (ind < n) {
            atomicAdd(&result[ptr[ind]], (data[ind])* b[index[ind]]);
        }
    }
    __syncthreads();

    for (int i = 0; i < numRow; i += ThreadBlockSize) {
        ind = i + threadIdx.x;
        if (ind < numRow) {
            t[ind] = result[ind];
        }
    }
}

int main(int argc, char **argv) {
    FILE *fp;
    char line[1024];
    int *ptr, *indices;
    float *data, *b, *t;
    int i, j;
    int n; // number of nonzero elements in data
    int numRow; // number of rows in matrix
    int numCol; // number of columns in matrix

                // Open input file and read to end of comments
    if (argc != 2) {
        fprintf(stderr, "No given parameter\n");
        exit(1);
    }

    if ((fp = fopen(argv[1], "r")) == NULL) {
        perror("Error");
        exit(1);
    }

    fgets(line, 128, fp);
    while (line[0] == '%') {
        fgets(line, 128, fp);
    }

    // Read number of rows (nr), number of columns (nc) and
    // number of elements and allocate memory for ptr, indices, data, b and t.
    sscanf(line, "%d %d %d\n", &numRow, &numCol, &n);
    size_t size_n = n * sizeof(int);
    size_t size_nRow = numRow * sizeof(float);
    size_t size_nCol = numCol * sizeof(float);
    ptr = (int *)malloc(size_n);
    indices = (int *)malloc(size_n);
    data = (float *)malloc(size_n);
    b = (float *)malloc(size_nCol);
    t = (float *)malloc(size_nRow);

    // Read data in coordinate format and initialize sparse matrix
    for (i = 0; i < n; i++) {
        fscanf(fp, "%d %d %f\n", &(ptr[i]), &(indices[i]), &(data[i]));
        indices[i]--;
        ptr[i]--;
    }

    // initialize t to 0 and b with random data  
    srand(time(NULL));

    for (i = 0; i<numRow; i++) {
        t[i] = 0.0;
    }

    for (i = 0; i<numCol; i++) {
        b[i] = (float)rand() / 1111111111;
    }

    // MAIN COMPUTATION, SEQUENTIAL VERSION
    for (i = 0; i < n; i++) {
        t[ptr[i]] += data[i] * b[indices[i]];
    }

    // This function: M*N   * N*1 = M*1
    //               (data)  (b)   (t)

    ///////////////Compute result on GPU and compare output/////////////////
    // allocate data on host
    int *h_ptr, *h_index;
    float *h_data, *h_b, *h_t;
    cudaHostAlloc(&h_ptr, size_n, cudaHostAllocMapped);
    cudaHostAlloc(&h_index, size_n, cudaHostAllocMapped);
    cudaHostAlloc(&h_data, size_n, cudaHostAllocMapped);
    cudaHostAlloc(&h_b, size_nCol, cudaHostAllocMapped);
    cudaHostAlloc(&h_t, size_nRow, cudaHostAllocMapped);

    // initialize host variables
    for (i = 0; i < n; i++) {
        h_index[i] = indices[i];
        h_data[i] = data[i];
        h_ptr[i] = ptr[i];
    }
    for (i = 0; i < numCol; i++) {
        h_b[i] = b[i];
    }
    for (i = 0; i < numRow; i++) {
        h_t[i] = 0;
    }

    // create pointers for device to use
    int *d_ptr, *d_index;
    float *d_data, *d_b, *d_t;
    cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
    cudaHostGetDevicePointer(&d_index, h_index, 0);
    cudaHostGetDevicePointer(&d_data, h_data, 0);
    cudaHostGetDevicePointer(&d_b, h_b, 0);
    cudaHostGetDevicePointer(&d_t, h_t, 0);

    // launch kernel
    SparseMatVecCUDA << <1, ThreadBlockSize, numRow * sizeof(float) >> >(d_ptr, d_index, d_data, d_b, d_t, n, numRow);
    cudaDeviceSynchronize();

    // compare cpu product and gpu product
    for (i = 0; i < numRow; i++) {
        if ((t[i] - h_t[i]) > 1e-10) {
            printf("Wrong calcuation, i=%d\n", i);
            printf("h_t = %f, while t = %f\n", h_t[i], t[i]);
            goto stop;
        }
    }
    printf("Comparsion complete, correct result\n");

stop:
    // free data
    cudaFreeHost(h_ptr);
    cudaFreeHost(h_index);
    cudaFreeHost(h_data);
    cudaFreeHost(h_b);
    cudaFreeHost(h_t);
    free(ptr);
    free(indices);
    free(data);
    free(b);
    free(t);
}
