#ifndef KACZMARZ_H
#define KACZMARZ_H

#include "linear_system.h"

void kaczmarz_solver(LinearSystem *sys, int max_iterations, double precision);

#endif // KACZMARZ_H
