#include <stdio.h>
#include "ft_blas.h"

/* To search for the optimized block size for matmul */

void randomize_matrix(gsl_matrix *A)
{
    int i, j;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, clock());

    for (i = 0; i < A->size1; i++) 
        for (j = 0; j < A->size2; j++) 
            gsl_matrix_set(A, i, j, gsl_ran_gaussian_ziggurat(r, 1));

    gsl_rng_free(r);
}
void outprod_dgemm(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C, int rank)
{
    int s;
    int mm, nn, k;
    gsl_matrix_view Av, Bv;

    mm = A->size1; nn = A->size2;
    k = A->size2;


    for (s = 0; s < k/rank * rank; s+=rank) {
        Av = gsl_matrix_submatrix(A, 0, s, mm, rank);
        Bv = gsl_matrix_submatrix(B, s, 0, rank, nn);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Av.matrix, &Bv.matrix, 1.0, C);
    }

    if (s < k) {
        Av = gsl_matrix_submatrix(A, 0, s, mm, k-s);
        Bv = gsl_matrix_submatrix(B, s, 0, k-s , nn);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Av.matrix, &Bv.matrix, 1.0, C);
    }
}
    
void find_opti_bsize()
{
    float t0, t1;
    gsl_matrix *A, *B, *C;
    int bs1=100, bs2=200, bs;
    int N=2000;
    
    A = gsl_matrix_alloc(N, N);
    B = gsl_matrix_alloc(N, N);
    C = gsl_matrix_alloc(N, N);

    randomize_matrix(A);
    randomize_matrix(B);


    /*printf("ATLAS performance:\n");*/
    /*t0 = clock();*/
    /*gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);*/
    /*t1 = clock();*/
    /*printf(" perf %f GFLOPS\n", */
           /*[>bs<]*/
            /*2* pow((N/1000.), 3) * CLOCKS_PER_SEC / (t1-t0)*/
          /*);*/
    float *ta;
    ta = (float*)calloc((bs2-bs1),sizeof(float));
    int n;
    double totalflops=0;
    for(n=1000; n<=N; n+=500){
        totalflops+=2.*n*n*n;
        A = gsl_matrix_alloc(n, n);
        B = gsl_matrix_alloc(n, n);
        C = gsl_matrix_alloc(n, n);
        randomize_matrix(A); randomize_matrix(B);

        for(bs = bs1; bs <= bs2; bs +=1) {
            t0 = clock();
            outprod_dgemm(A, B, C, bs);
            t1 = clock();
            ta[bs-bs1] += t1-t0;
        }
        gsl_matrix_free(A);gsl_matrix_free(B);gsl_matrix_free(C);
    }
    int best_bs = bs1;
    int best_bs_t = ta[0];
    for(bs=bs1;bs<=bs2;bs++)
        if(ta[bs-bs1] < best_bs_t) {
            best_bs_t = ta[bs-bs1];
            best_bs = bs;
        }
    printf("//best BS betwee %d and %d is %d\n", bs1,bs2, best_bs);
    printf("#define MMRANK %d\n", best_bs);
    printf("//matmul performance in FLOPS\n");
    printf("#define MMFLOPS %f\n", totalflops*CLOCKS_PER_SEC/best_bs_t);
    free(ta);

}
/*void find_matmul_gflops()*/
/*{*/
    /*float t0, t1;*/
    /*gsl_matrix *A, *B, *C;*/
    /*int bs1=100, bs2=200, bs;*/
    /*int N=3000;*/
    
    /*A = gsl_matrix_alloc(N, N);*/
    /*B = gsl_matrix_alloc(N, N);*/
    /*C = gsl_matrix_alloc(N, N);*/

    /*randomize_matrix(A);*/
    /*randomize_matrix(B);*/

int main(int argc, char *argv[])
{
    find_opti_bsize();
    return 0;
}
