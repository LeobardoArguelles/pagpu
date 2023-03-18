
// Function in cuda to multiply two matrices of different sizes
// and store the result in a third matrix
// The matrices are stored in row-major order
__global__ void mmultipy(float *A, float *B, float *C, int m, int n, int p)
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
