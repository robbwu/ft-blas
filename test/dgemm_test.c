#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#ifndef GSL_RANGE_CHECK_OFF
#define GSL_RANGE_CHECK_OFF
#endif
long  N,ER;
gsl_matrix *target;
void print_matrix(gsl_matrix *A)
{
    int i, j;

    for (i = 0; i < A->size1; i++) {
        for (j = 0; j < A->size2; j++) {
            printf("%g\t", gsl_matrix_get(A, i, j));
        }
        printf("\n");
    }
    printf("\n");
}
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
void* noise_thread(void *vargp)
{
    gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, clock());
    int i, j;
    double mag;
    long int msecs;

    pthread_setcanceltype(PTHREAD_CANCEL_ENABLE, NULL);

    usleep(500);
    i = floor(gsl_rng_uniform(r) * (N-1));
    j = floor(gsl_rng_uniform(r) * (N-1));
    mag = gsl_ran_gaussian_ziggurat(r, 10);
    /*printf("C[%d, %d] = %f  ", i, j, gsl_matrix_get(target, i, j));*/
    gsl_matrix_set(target, i, j, gsl_matrix_get(target, i, j)* mag);
    printf("simulated corruption!\n");

    while(1) { 
        msecs = floor(0.5 * N/1000.0 * N * N / ER * 
                (1+fabs(gsl_ran_gaussian_ziggurat(r, 0.1))));
        /*msecs = floor(0.8 * N/1000. * N * N / ER);*/
        /*printf("secs: %f\n", msecs/1.e6);*/
        usleep(msecs);
        printf("simulated corruption!\n");
        i = floor(gsl_rng_uniform(r) * (N-1));
        j = floor(gsl_rng_uniform(r) * (N-1));
        mag = gsl_ran_gaussian_ziggurat(r, 10);
        /*printf("C[%d, %d] = %f  ", i, j, gsl_matrix_get(target, i, j));*/
        gsl_matrix_set(target, i, j, gsl_matrix_get(target, i, j)* mag);
        /*printf("noise: C[%d, %d] = %.3f\n", i, j, mag*gsl_matrix_get(target, i, j)); */
    }

    gsl_rng_free(r);
    return NULL;
}
void test_ft_dgemm()
{

    float t0, t1;
    int m, n, k, s;
    gsl_matrix *A, *B, *C, *D;

    double infnorm(gsl_matrix *A);

    A = gsl_matrix_alloc(N, N);
    B = gsl_matrix_alloc(N, N);
    C = gsl_matrix_alloc(N, N);
    D = gsl_matrix_alloc(N, N);
    target = C;
    m = A->size1; n = B->size2; k = A->size2;
    randomize_matrix(A);
    randomize_matrix(B);
    /*row_sums = calloc(C->size1, sizeof(double));*/
    /*col_sums = calloc(C->size2, sizeof(double));*/

    build_checksum(A, B, C);

    pthread_t tid;
    pthread_create(&tid, NULL, noise_thread, NULL);


    s = floor(N/ER);
    t0 = clock();
    ft_dgemm(A, B, C, s);
    t1 = clock();
    pthread_cancel(tid);
    printf("ft_dgemm: %.3fGFLOPS %.3fs\n", 
            2* pow((N/1000.), 3) * CLOCKS_PER_SEC / (t1-t0), (t1-t0)/CLOCKS_PER_SEC);

    t0 = clock();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, D);
    t1 = clock();
    printf("dgemm: %.3fGFLPS %.3fs\n", 
            2* pow((N/1000.), 3) * CLOCKS_PER_SEC / (t1-t0), (t1-t0)/CLOCKS_PER_SEC);
    gsl_matrix_sub(C,D); 
    printf("error(infnorm): %f\n", infnorm(C));

    pthread_create(&tid, NULL, noise_thread, NULL);
    t0 = clock();
    ft_dgemm(A, B, C, N);
    t1 = clock();
    printf("huang_dgemm: %.3fGFLOPS %.3f\n", 
            2* pow((N/1000.), 3) * CLOCKS_PER_SEC / (t1-t0), (t1-t0)/CLOCKS_PER_SEC);
    pthread_cancel(tid);
    gsl_matrix_sub(C,D); 
    printf("error(infnorm): %f\n", infnorm(C));


    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
    gsl_matrix_free(D);
}
int main(int argc, char *argv[])
{
    if (argc < 3){ 
        printf("Usage: %s matrix_size errors\n", argv[0]);
        exit(1);
    }

    N = atoi(argv[1]);
    ER = atoi(argv[2]);
    test_ft_dgemm();
}
