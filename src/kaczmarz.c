#include <math.h>
#include <stdio.h>
#include "kaczmarz.h"

void kaczmarz_solver(LinearSystem *sys, int n) {
    for (int iter = 0; iter < n; iter++) {
        for (int i = 0; i < sys->rows; i++) {
            double dot_product = 0.0;
	    double a_norm = 0.0;
            for (int j = 0; j < sys->cols; j++) {
                dot_product += sys->A[i][j] * sys->x[j];
		a_norm += sys->A[i][j]*sys->A[i][j]; 
	    }
	    if (a_norm < 1e-10){
		    printf("Matrix column with 0 norm, iteration not possible");
		    return;
	    }
	    double correction = (sys->b[i] - dot_product) / (a_norm);
            for (int j = 0; j < sys->cols; j++) {
                sys->x[j] += sys->A[i][j] * correction;
            }
        }
    }
}
