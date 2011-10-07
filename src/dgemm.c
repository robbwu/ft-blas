/* USE AT YOUR OWN RISKS!!
 * Author: Panruo Wu(armiusuw@gmail.com)
 * Last Modified Date: 06/10/2011
 */

/*#include <stdio.h>*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#ifndef GSL_RANGE_CHECK_OFF
#define GSL_RANGE_CHECK_OFF
#endif
long FR;
double *row_sums, *col_sums, all_sum;
double u = 1.1e-16;     // unit roundoff error of IEEE double
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
double vec_sum(gsl_vector *v)
{
    double sum = 0;
    int i;

    for (i = 0; i < v->size; i+=2) 
        sum += (gsl_vector_get(v, i) + gsl_vector_get(v, i+1));
    
    return sum;
}
void build_checksum(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C)
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

    gsl_matrix_set_zero(C);

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

    memset((void*)row_sums, 0, sizeof(double)*(m-1));
    memset((void*)col_sums, 0, sizeof(double)*(n-1));

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
            /*printf("row_sums[ii] = %f, C[ii, jj] = %f\n" , */
                    /*row_sums[ii], gsl_matrix_get(C, ii, jj));*/
            gsl_matrix_set( C, ii, jj, -row_sums[ii] + gsl_matrix_get(C, ii, jj));
            return 1;
        }
    } else if (in == 0  && jn == 0) 
        return 0;
    else 
        return -2;
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

    lambda = infnorm(A) * infnorm(B) * k * u;
    mu = onenorm(A) * onenorm(B) * k * u;
    tau = (lambda < mu) ? lambda : mu;


    // build checksum matrices A^c, B^r
retry:
    trials++;
    build_checksum(A, B, C);

    
    for (s = 0; s < k/rank * rank; s+=rank) {
        err = verify_checksum(C);
        if (err < 0 && trials < 5)
            goto retry;
        Av = gsl_matrix_submatrix(A, 0, s, mm, rank);
        Bv = gsl_matrix_submatrix(B, s, 0, rank, nn);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Av.matrix, &Bv.matrix, 1.0, C);
    }

    if (s < k) {
        err = verify_checksum(C);
        /*printf("err: %d\n", err); */
        if (err < 0 && trials < 5)
            goto retry;
        Av = gsl_matrix_submatrix(A, 0, s, mm, k-s);
        Bv = gsl_matrix_submatrix(B, s, 0, k-s , nn);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &Av.matrix, &Bv.matrix, 1.0, C);
    }
    free(row_sums); free(col_sums);
    return trials;

}
