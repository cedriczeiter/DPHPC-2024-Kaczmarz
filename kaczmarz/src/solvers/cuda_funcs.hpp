#ifndef CUDA_FUNCS_HPP
#define CUDA_FUNCS_HPP

void dot_product_cuda(const double *h_A, const double *h_x, double *h_result, const unsigned rows, const unsigned cols);

void squared_norm_cuda(const double *h_A, double *h_result, const unsigned rows, const unsigned cols);

#endif  // CUDA_FUNCS_HPP