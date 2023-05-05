#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include<cuda_runtime_api.h>
#include<time.h>
#include<locale.h>


#define T_WIDTH 32

__global__ void mmultiplyKernel_shared(float* M, float* N, float* P, int m, int n, int p) {
    __shared__ float Mds[T_WIDTH][T_WIDTH];
    __shared__ float Nds[T_WIDTH][T_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    // Each thread works on one element of P
    // by taking one tile row of M and one tile column of N
    int Row = by * T_WIDTH + ty;
    int Col = bx * T_WIDTH + tx;

    float Pvalue = 0;

    // Loop over the M and N tiles required to compute the P element
   for (int k = 0; k < (n-1)/T_WIDTH+1; ++k) {

        // Collaborative loading of K and N tiles into shared memory
        if (Row < m && k*T_WIDTH+tx < n)
            Mds[ty][tx] = M[Row*n + k*T_WIDTH+tx];
        else
            Mds[ty][tx] = 0.0;

        if (k*T_WIDTH+ty < n && Col < p)
            Nds[ty][tx] = N[(k*T_WIDTH+ty)*p + Col];
        else
            Nds[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < T_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    if (Row < m && Col < p)
        P[Row*p+Col] = Pvalue;
}

__global__ void mmultiplyKernel_global(float* A, float* B, float* C, int m, int n, int p) //Kernel dentro del GPU
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

__host__ void vecMultiply(float* h_A, float* h_B, float* h_C, int m, int n, int p, bool shared = 0) //Kernel dentro del GPU
{
    float* d_A, * d_B, * d_C; //Device en GPU

    // Device memory allocation
    check(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
    check(cudaMalloc((void**)&d_B, n * p * sizeof(float)));
    check(cudaMalloc((void**)&d_C, m * p * sizeof(float)));

    // Copy values from host to device
    check(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize grid dimensions
    printf("(%d, %d)\n", (p-1)/T_WIDTH+1, (m-1)/T_WIDTH+1);
    dim3 DimGrid((p-1)/T_WIDTH+1, (m-1)/T_WIDTH+1, 1);
    dim3 DimBlock(T_WIDTH, T_WIDTH, 1);

    //dim3 DimGrid((m - 1) / 32 + 1, (p - 1) / 32 + 1);
    //dim3 DimBlock(32, 32);

    // Call kernel
    if (shared == 1) {
      mmultiplyKernel_shared << < DimGrid, DimBlock >> > (d_A, d_B, d_C, m, n, p);
    }
    else {
        mmultiplyKernel_global << < DimGrid, DimBlock >> > (d_A, d_B, d_C, m, n, p);
    }
    cudaDeviceSynchronize();

    // Copy result
    check(cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory in device
    check(cudaFree(d_A));
    check(cudaFree(d_B));
    check(cudaFree(d_C));
}

__host__ void showMatrix(float *matrix, int m, int p) {
    int i, j;
    int size = m*p;
    int last = size - 1;

    if (m < 10 && p < 10) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < p; j++) {
                printf("%0.2f\t", matrix[i*p+j]);
            }
            printf("\n");
        }
    } else {
        for (i = 0; i < 5; i++) {
            for (j = 0; j < 5; j++) {
                printf("%0.2f\t", matrix[i*p+j]);
            }
            printf("\n");
        }
        printf("...\n");
        for (i = 4; i >= 0; i--) {
            for (j = 4; j >= 0; j--) {
                printf("%0.2f\t", matrix[last-i*p-j]);
            }
            printf("\n");
        }
    }
}

__host__ void calcTime(clock_t start, clock_t end) {
    double duration = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Time: %f\n", duration);
}

int compare_floats(float A, float B) {
    if (abs(A - B) > 1e-1)
        return 1;
    else
        return 0;
}

__host__ bool check_results(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        if (!(compare_floats(A[i], B[i]) == 0 && compare_floats(B[i], C[i]) == 0)) {
          printf("[%d] -> %0.20f %0.20f %0.20f\n", i, A[i], B[i], C[i]);
          return false;
    }
    }
    return true;
}

int main()
{
    int m, n, p;
    clock_t s, f;

    printf("Ingrese el tamaño de la matriz A (m x n): ");
    scanf("%d %d", &m, &n);

    printf("Ingrese el tamaño de la matriz B (%d x p): ", n);
    scanf("%d", &p);

    float** A = (float**)malloc(m * sizeof(float*));
    for (int i = 0; i < m; i++)
        A[i] = (float*)malloc(n * sizeof(float));

    float** B = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++)
        B[i] = (float*)malloc(p * sizeof(float));

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

    float* h_A = (float*)malloc(m * n * sizeof(float));
    float* h_B = (float*)malloc(n * p * sizeof(float));
    float* h_C_global = (float*)malloc(m * p * sizeof(float));
    float* h_C_shared = (float*)malloc(m * p * sizeof(float));
    float* h_C_local = (float*)malloc(m * p * sizeof(float));

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

    printf("GPU - Global memory\n");

    s = clock();

    vecMultiply(h_A, h_B, h_C_global, m, n, p);

    f = clock();

    calcTime(s, f);

    printf("GPU - Shared memory\n");

    s = clock();

    vecMultiply(h_A, h_B, h_C_shared, m, n, p, 1);

    f = clock();

    calcTime(s, f);

    printf("CPU\n");

    s = clock();

    mmultiply_Sec(h_A, h_B, h_C_local, m, n, p);

    f = clock();

    calcTime(s, f);
    if (!check_results(h_C_global, h_C_shared, h_C_local, m*p)) {
        printf("Los resultados no coinciden");
        return 1;
    }
    printf("\nMatriz A:\n");
    showMatrix(h_A, m, n);
    printf("\nMatriz B:\n");
    showMatrix(h_B, n, p);
    printf("\nMatriz C:\n");
    showMatrix(h_C_local, m, p);

    free(A);
    free(B);
    free(h_A);
    free(h_B);
    free(h_C_local);
    free(h_C_global);
    free(h_C_shared);

    return 0;
}
