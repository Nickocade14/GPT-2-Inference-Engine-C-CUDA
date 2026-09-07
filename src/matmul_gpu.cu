#include <cstdio> 
void matrixA_Filled(float *matrixA, int M, int K){
    for(int i = 0; i < M*K; i++){
        matrixA[i] = float(i + 1);
    }
}
void matrixB_Filled(float *matrixB, int K, int N){
    for(int i = 0; i < K*N; i++){
        matrixB[i] = float(i + 1);
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
    
    // Device Varibles
    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, (M * K) *sizeof(float));
    cudaMalloc(&d_B, (K * N) *sizeof(float));
    cudaMalloc(&d_C, (M * N) *sizeof(float));

    //Host Variables 
    
    float *matrixA = new float[M * K];
    float *matrixB = new float[K * N];
    float *matrixC = new float[M * N];

    matrixA_Filled(matrixA, M, K);
    matrixB_Filled(matrixB, K, N);

    // Copy the host matrices into the device matrices
    cudaMemcpy(d_A, matrixA, (M*K) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matrixB, (K*N) * sizeof(float), cudaMemcpyHostToDevice);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;
}