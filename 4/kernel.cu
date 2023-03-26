#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<cuda_runtime_api.h>
#include<time.h>
#include<locale.h>

__global__ void mmultiplyKernel(float* A, float* B, float* C, int m, int n, int p) //Kernel dentro del GPU
{
    // Calculate the row index of the C element and return if it is out of bounds
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m)
        return;

    // Calculate the column index of the C element and return if it is out of bounds
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= p)
        return;

    // Each thread computes one element of the block sub-matrix
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += A[row * n + i] * B[i * p + col];
    C[row * p + col] = sum;

}

__host__ void mmultiply_Sec(float* A, float* B, float* C, int m, int n, int p)
{
    int i, j, k;
    float sum = 0;

    // Multiply the matrices
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < p; j++)
        {
            for (k = 0; k < n; k++)
            {
                sum = sum + A[i * n + k] * B[k * p + j];
            }

            C[i * p + j] = sum;
            sum = 0;
        }
    }

}

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__host__ void vecMultiply(float* h_A, float* h_B, float* h_C, int m, int n, int p) //Kernel dentro del GPU
{
    float *d_A, *d_B, *d_C; //Device en GPU 

    // Device memory allocation
    check(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
    check(cudaMalloc((void**)&d_B, n * p * sizeof(float)));
    check(cudaMalloc((void**)&d_C, m * p * sizeof(float)));

    // Copy values from host to device
    check(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize grid dimensions
    dim3 DimGrid((m - 1) / 32 + 1, (p - 1) / 32 + 1);
    dim3 DimBlock(32, 32);

    // Call kernel
    mmultiplyKernel << < DimGrid, DimBlock >> > (d_A, d_B, d_C, m, n, p); 
    cudaDeviceSynchronize();

    // Copy result
    check(cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory in device
    check(cudaFree(d_A));
    check(cudaFree(d_B));
    check(cudaFree(d_C));
}

__host__ void showMatrix(float* vector, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%-9.2f", vector[i * m + j]);
        }
        printf("\n");
    }
}

__host__ void calcTime(clock_t start, clock_t end) {
    double duration = ((double)end - start) / CLOCKS_PER_SEC;
    printf("\nTime: %f", duration);
}

int main()
{
    setlocale(LC_ALL, "");
    int m, n, p;
    clock_t s, f;

    printf("Ingrese el tamaño de la matriz A (m x n): ");
    scanf("%d %d", &m, &n);

    printf("Ingrese el tamaño de la matriz B (%d x p): ", n);
    scanf("%d", &p);

    float** A = (float**)malloc(m * sizeof(float*));
    for (int i = 0; i < m; i++)
        A[i] = (float*)malloc(m * sizeof(float));

    float** B = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++)
        B[i] = (float*)malloc(p * sizeof(float));

    int automatic;
    int metodo;
    printf("Desea llenar las matrices de manera automtica (1) o manual (0)? ");
    scanf("%d", &automatic);

    if (automatic == 0) {
        printf("Ingrese los elementos de la matriz A:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%f", &A[i][j]);
            }
        }

        printf("Ingrese los elementos de la matriz B:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                scanf("%f", &B[i][j]);
            }
        }
    }
    else {
        srand(time(0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = (float)rand() / RAND_MAX * 9.0 + 1.0;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                B[i][j] = (float)rand() / RAND_MAX * 9.0 + 1.0;
            }
        }
    }

    printf("Deseas procesar la matrices de manera CPU (1) o GPU (0)? ");
    scanf("%d", &metodo);

    float* h_A = (float*)malloc(m * n * sizeof(float));
    float* h_B = (float*)malloc(n * p * sizeof(float));
    float* h_C = (float*)malloc(m * p * sizeof(float));

    // Line compression

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++) {
            h_A[(i * n) + j] = A[i][j];
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < p; j++) {
            h_B[(i * p) + j] = B[i][j];
        }
    }

    if (metodo == 0) {

        s = clock();

        vecMultiply(h_A, h_B, h_C, m, n, p);

        f = clock();

        calcTime(s, f);

    }
    else {

        s = clock();

        mmultiply_Sec(h_A, h_B, h_C, m, n, p);

        f = clock();

        calcTime(s, f);
    }

    printf("\nMatriz A:\n");
    showMatrix(h_A, m, n);
    printf("\nMatriz B:\n");
    showMatrix(h_B, n, p);
    printf("\nMatriz C:\n");
    showMatrix(h_C, m, p);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

