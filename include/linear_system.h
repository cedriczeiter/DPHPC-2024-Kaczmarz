#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

typedef struct {
  double **A; // Matrix A
  double *b;  // Vector b
  double *x;  // Solution vector x
  int rows, cols;
} LinearSystem;

LinearSystem *allocate_system(int rows, int cols);
void deallocate_system(LinearSystem *sys);

#endif // LINEAR_SYSTEM_H