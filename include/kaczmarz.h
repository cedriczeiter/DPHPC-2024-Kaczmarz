#ifndef KACZMARZ_H
#define KACZMARZ_H

#include "linear_system.h"

void kaczmarz_solver(LinearSystem *sys, int max_iterations, double precision);

int random_row_selection(double *row_norms, int num_rows);
void kaczmarz_random_solver(LinearSystem *sys, int max_iterations, double precision);

#endif // KACZMARZ_H
