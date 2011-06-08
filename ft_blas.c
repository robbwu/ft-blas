#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#ifndef GSL_RANGE_CHECK_OFF
#define GSL_RANGE_CHECK_OFF
#endif
#define N 500
double *row_sums, *col_sums, all_sum;
double u = 2.2e-16;
double lambda, mu, tau;  // thresholds for distinguishing roundoff between fault

double infnorm(gsl_matrix *A)
{
    int m = A->size1, n = A->size2;
    int i, j;
    double norm = 0.0, d;

    gsl_vector_view row;
    for (i = 0; i < m; i++) {
        row = gsl_matrix_row(A, i);
        d = gsl_blas_dasum(&row.vector);
        norm = (d < norm) ? norm : d;
    }
    return norm;
}
double onenorm(gsl_matrix *A)
{
    int m = A->size1, n = A->size2;
    int i, j;
    double norm = 0.0, d;

    gsl_vector_view col;
    for (i = 0; i < n; i++) {
        col = gsl_matrix_column(A, i);
        d = gsl_blas_dasum(&col.vector);
        norm = (d < norm) ? norm : d;
    }
    return norm;

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
double vec_sum(gsl_vector *v)
{
    double sum = 0;
    int i;

    for (i = 0; i < v->size; i+=2) 
        sum += (gsl_vector_get(v, i) + gsl_vector_get(v, i+1));
    
    return sum;
}
void build_checksum(gsl_matrix *A, gsl_matrix *B)
{
    gsl_vector_view sum, v;
    int i;
    int m = A->size1 - 1, n = B->size2 -1;

    sum = gsl_matrix_row(A, m);
    gsl_vector_set_zero(&sum.vector);

    for (i = 0; i < m; i++) {
         v = gsl_matrix_row(A, i);
         gsl_vector_add(&sum.vector, &v.vector);
    }

    sum = gsl_matrix_column(B, n);
    gsl_vector_set_zero(&sum.vector);
    for (i = 0; i < n; i++) {
        v = gsl_matrix_column(B, i);
        gsl_vector_add(&sum.vector, &v.vector);
    }

}
int ft_dgemm(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C, int rank)
{
    int err, trials= 0;
    int i, j, s;
    gsl_matrix_view Av, Bv;
    int m, n, k;
    int mm, nn;

    m = A->size1 - 1;
    n = B->size2 - 1;
    k = A->size2;
    mm = m+1; nn = n+1;
    
    // allocate workspace for verify_checksum
    row_sums = calloc(m, sizeof(double));
    col_sums = calloc(n, sizeof(double));

    // build checksum matrices A^c, B^r
retry:
    trials++;
    build_checksum(A, B);

    
    /*printf("[ft_dgemm]: (m, k, n) = (%d, %d, %d)\n", m, k, n);*/
    for (s = 0; s < k/rank * rank; s+=rank) {
        gsl_matrix_set(C, 0, 0, 2010.0);
        err = verify_checksum(C);
        /*printf("err: %d", err); */
        if (err < 0 && trials < 5)
            goto retry;
        Av = gsl_matrix_submatrix(A, 0, s, mm, rank);
        Bv = gsl_matrix_submatrix(B, s, 0, rank, nn);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Av.matrix, &Bv.matrix, 1.0, C);
    }

    if (s < k) {
        err = verify_checksum(C);
        if (err < 0 && trials < 5)
            goto retry;
        Av = gsl_matrix_submatrix(A, 0, s, mm, k-s);
        Bv = gsl_matrix_submatrix(B, s, 0, k-s , nn);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Av.matrix, &Bv.matrix, 1.0, C);
    }
    return trials;

}
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
/* return value :
 * -2 : failure, too many faults
 * -1 : failure, unable to recover
 *  0 : pass
 *  1 : failure, recovered
 * */
int verify_checksum(gsl_matrix *C)
{
    int i, j, ii, jj, in, jn;
    int m = C->size1, n = C->size2;
    double c;

    // calculate row & col sums.
    for (i = 0; i < m-1; i++) {
        for (j = 0; j < n-1; j++) {
            c = gsl_matrix_get(C, i, j);
            row_sums[i] += c;
            col_sums[j] += c;
        }
    }
    
    //verify checksum
    in = jn = 0;
    for (i = 0; i < m-1; i++) {
        row_sums[i] -= gsl_matrix_get(C, i, n-1);
        if( fabs(row_sums[i]) > lambda ){
            in++;
            ii = i;
        }
    }
    for (j = 0; j < n-1; j++) {
        col_sums[j] -= gsl_matrix_get(C, m-1, j);
        if( fabs(col_sums[j]) > mu){
            jn++;
            jj= j;
        }
    }

    if (in == 1 && jn == 1) {
        if ( fabs(row_sums[ii] - col_sums[jj]) > tau )
            return -1;
        else {
            gsl_matrix_set( C, ii, jj, -row_sums[ii] + gsl_matrix_get(C, i, j));
            return 1;
        }
    } else if (in == 0  && jn == 0) 
        return 0;
    else 
        return -2;
}

void test_ft_dgemm()
{

    float t0, t1;
    int m, n, k;
    gsl_matrix *A, *B, *C;
    A = gsl_matrix_alloc(N, N);
    B = gsl_matrix_alloc(N, N);
    C = gsl_matrix_alloc(N, N);
    m = A->size1; n = B->size2; k = A->size2;
    randomize_matrix(A);
    randomize_matrix(B);
    row_sums = calloc(C->size1, sizeof(double));
    col_sums = calloc(C->size2, sizeof(double));

    build_checksum(A, B);
    lambda = infnorm(A) * infnorm(B) * k * u;
    mu = onenorm(A) * onenorm(B) * k * u;
    tau = (lambda < mu) ? lambda : mu;

    t0 = clock();
    ft_dgemm(A, B, C, 200);
    t1 = clock();
    printf("ft_dgemm: %.3f\n", 2* pow((N/1000.), 3) * CLOCKS_PER_SEC / (t1-t0));

    t0 = clock();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
    t1 = clock();
    printf("dgemm: %.3f\n", 2* pow((N/1000.), 3) * CLOCKS_PER_SEC / (t1-t0));

    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
    free(row_sums);free(col_sums);
}
int main(int argc, char *argv[])
{
    test_ft_dgemm();
}
