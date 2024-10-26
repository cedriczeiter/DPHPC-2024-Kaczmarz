#ifndef UTILS_HPP
#define UTILS_HPP

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

int export_matrix(Eigen::SparseMatrix<double> A, std::string path) {
  const unsigned rows = A.rows();
  const unsigned cols = A.cols();
  std::ofstream outFile(path);
  if (!outFile.is_open()) {
    std::cerr << "Error opening file for writing!" << std::endl;
    return 1;
  }
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      outFile << it.row() << " " << it.col() << " " << it.value() << std::endl;
    }
  }
  return 0;
}

#endif
