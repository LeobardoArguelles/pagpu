#include <stdio.h>
#include <stdlib.h>


// Function to multiply two matrices of different sizes
// and store the result in a third matrix
// A: first matrix
// B: second matrix
// C: result matrix
// m: number of rows in A
// n: number of columns in A and number of rows in B
// p: number of columns in B
// return: pointer to the result matrix
float *mmultiply(float *A, float *B, float *C, int m, int n, int p)
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

    return C;
}
