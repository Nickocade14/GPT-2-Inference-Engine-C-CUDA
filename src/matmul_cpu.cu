#include <cstdio> 

void assign_matrixC(float *matrixA, float *matrixB, float *matrixC, int M, int K, int N){ 
    for (int i = 0; i < M; i++){ 
        for (int j = 0; j < N; j++){ 
            float sum = 0; 
            for (int k = 0; k < K; k++){ 
                sum += (matrixA[(i * K) + k]) * (matrixB[k * N + j]);  
            } 
            matrixC[(i * N) + j] = sum; 
        } 
    } 
} 
 
int main(){ 
    /* 
    Matrix A = (M, K) 
    Matrix B = (K, N) 
    Matrix C = (M, N) 
    */ 
    int M = 3; 
    int K = 4; 
    int N = 2; 
 
    // Allocating memory for matrices A, B, and C
    float *matrixA = new float[M * K]; 
    float *matrixB = new float[K * N]; 
    float *matrixC = new float[M * N]; 
 
    // Assigning values to Matrix A and Matrix B
    for (int i = 0; i < M*K; i++){ 
        matrixA[i] = float(i + 1); 
    } 
     
    for (int j = 0; j < K*N; j++){ 
        matrixB[j] = float(j + 1); 
    } 
 
    // Assigning values to Matrix C based on the result of Matrix A and Matrix B multiplication
    assign_matrixC(matrixA, matrixB, matrixC, M, K, N); 

    for (int k = 0; k < M*N; k++){ 
        printf("%.2f ", matrixC[k]); 
    } 
 
    // Freeing the memory allocated for the three matrices
    delete[] matrixA; 
    delete[] matrixB; 
    delete[] matrixC;
}