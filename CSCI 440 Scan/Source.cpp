#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

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