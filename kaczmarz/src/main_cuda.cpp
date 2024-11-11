#include <iostream>
#include <cassert>

void squared_norm_cuda(const double *h_A, double *h_result, const unsigned rows, const unsigned cols);
void dot_product_cuda(const double *h_A, const double *h_x, double *h_result, const unsigned rows, const unsigned cols);

void test_squared_norm_cuda() {
  const unsigned rows = 2, cols = 3;
  double h_A[rows * cols] = {1, 2, 3, 4, 5, 6};
  double h_result[rows], expected[rows] = {14, 77};

  squared_norm_cuda(h_A, h_result, rows, cols);

  for (unsigned i = 0; i < rows; i++) {
    assert(h_result[i] == expected[i]);
  }

  std::cout << "Test for squared_norm_cuda passed!" << std::endl;
}

void test_dot_product_kernel() {
  const unsigned rows = 2, cols = 3;
  double h_A[rows * cols] = {1, 2, 3, 4, 5, 6};
  double h_x[cols] = {1, 1, 1};
  double h_result[rows], expected[rows] = {6, 15};

  dot_product_cuda(h_A, h_x, h_result, rows, cols);

  for (unsigned i = 0; i < rows; i++) {
    assert(h_result[i] == expected[i]);
  }

  std::cout << "Test for dot_product_kernel passed!" << std::endl;
}

int main() {
    test_dot_product_kernel();
    test_squared_norm_cuda();
    return 0;
}