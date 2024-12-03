#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;

std::string get_output_filename(const std::string &input_filename) {
  size_t last_dot = input_filename.find_last_of(".");
  if (last_dot == std::string::npos) {
    return input_filename + "_banded";
  } else {
    return input_filename.substr(0, last_dot) + "_banded" +
           input_filename.substr(last_dot);
  }
}

void export_matrix(const Eigen::SparseMatrix<double> &matrix,
                   const Eigen::VectorXd &rhs, const std::string &filename) {
  std::ofstream out_stream(filename);
  if (!out_stream.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  out_stream << matrix.nonZeros() << std::endl;
  out_stream << matrix.rows() << " " << matrix.cols() << std::endl;

  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
      out_stream << it.row() << " " << it.col() << " " << it.value()
                 << std::endl;
    }
  }

  for (int i = 0; i < rhs.size(); ++i) {
    out_stream << rhs[i] << std::endl;
  }

  out_stream.close();
}

void write_sparsity_pattern(const Eigen::SparseMatrix<double> &matrix,
                            const std::string &filename) {
  std::ofstream out_stream(filename);
  if (!out_stream.is_open()) {
    std::cerr << "Error: Could not open output file: " << filename << std::endl;
    return;
  }

  // Write each non-zero element's row and column index to the file
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
      out_stream << it.row() << " " << it.col() << "\n";
    }
  }

  out_stream.close();
}

int compute_bandwidth(const Eigen::SparseMatrix<double> &A) {
  int bandwidth = 0;

  // Traverse each row (or column) of the sparse matrix
  for (int i = 0; i < A.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
      int row = it.row();  // Row index of the current nonzero entry
      int col = it.col();  // Column index of the current nonzero entry
      bandwidth = std::max(bandwidth, std::abs(row - col));
    }
  }

  return bandwidth;
}

std::vector<int> reverse_cuthill_mckee(const Eigen::SparseMatrix<double> &A) {
  int n = A.rows();
  std::vector<int> perm(n, -1);  // Output permutation (will store node indices)
  std::vector<int> degree(n, 0);        // Degree of each node
  std::vector<bool> visited(n, false);  // Visited nodes

  // Step 1: Compute degree of each node
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      if (it.row() != it.col()) {  // Ignore self-loops
        degree[it.row()]++;
        degree[it.col()]++;
      }
    }
  }

  int index = 0;  // Result permutation index

  // Step 2: Perform BFS for all nodes to ensure all components are covered
  while (index < n) {
    // Find an unvisited node with the smallest degree to start a new BFS
    int start_node = -1;
    int min_degree = n + 1;  // Start with a large degree
    for (int j = 0; j < n; ++j) {
      if (!visited[j] && degree[j] < min_degree) {
        min_degree = degree[j];
        start_node = j;
      }
    }

    if (start_node == -1) break;  // All nodes have been visited

    // Step 3: Start BFS from `start_node`
    std::queue<int> Q;
    Q.push(start_node);
    visited[start_node] = true;

    while (!Q.empty()) {
      int v = Q.front();
      Q.pop();
      perm[index++] = v;  // Assign current node to perm

      // Collect unvisited neighbors
      std::vector<int> neighbors;
      for (Eigen::SparseMatrix<double>::InnerIterator it(A, v); it; ++it) {
        int neighbor = it.col();
        if (!visited[neighbor]) {
          neighbors.push_back(neighbor);
          visited[neighbor] = true;
        }
      }

      // Sort neighbors by degree (ascending order)
      std::sort(neighbors.begin(), neighbors.end(),
                [&degree](int a, int b) { return degree[a] < degree[b]; });

      // Add sorted neighbors to the queue
      for (int neighbor : neighbors) {
        Q.push(neighbor);
      }
    }
  }

  // Reverse the permutation for RCM order
  std::reverse(perm.begin(), perm.end());

  /*
  // Debugging: Print permutation vector to check correctness
  std::cout << "Permutation vector (perm): ";
  for (int p : perm) {
    std::cout << p << " ";
  }
  std::cout << std::endl;
  */

  // Check if the perm vector is fully populated
  for (int p : perm) {
    if (p == -1) {
      std::cerr
          << "Error: Permutation vector contains unvisited nodes (-1 values)."
          << std::endl;
      break;
    }
  }

  return perm;
}

struct SparseLinearSystem {
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd b;
};

SparseLinearSystem reorder_system_rcm(const SparseLinearSystem &system) {
  std::vector<int> perm = reverse_cuthill_mckee(system.A);

  // Check if the perm vector has the correct size
  if (perm.size() != system.A.rows()) {
    std::cerr
        << "Error: Permutation vector size does not match matrix row count!"
        << std::endl;
    return system;  // Return the original system in case of an error
  }

  Eigen::VectorXi perm_eigen =
      Eigen::Map<Eigen::VectorXi>(perm.data(), perm.size());
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_matrix(
      perm_eigen);

  // Permute the system
  Eigen::SparseMatrix<double> A_temp =
      perm_matrix.transpose() * system.A * perm_matrix;
  Eigen::VectorXd b_reordered = perm_matrix.transpose() * system.b;

  SparseLinearSystem reordered_system;
  reordered_system.A = A_temp;
  reordered_system.b = b_reordered;
  return reordered_system;
}

int main() {
  std::string input_filename = "../problem3_complexity6.txt";
  std::ifstream in_stream(input_filename);
  if (!in_stream.is_open()) {
    std::cerr << "Error: Could not open input file." << std::endl;
    return 1;
  }

  unsigned nnz, rows, cols;
  in_stream >> nnz >> rows >> cols;
  // std::cout << nnz << std::endl;
  if (in_stream.fail() || nnz == 0 || rows == 0 || cols == 0) {
    std::cerr << "Error: Invalid file format or empty input." << std::endl;
    return 1;
  }

  std::vector<Eigen::Triplet<double>> triplets_A;
  triplets_A.reserve(nnz);

  // Read triplets for the sparse matrix
  for (unsigned i = 0; i < nnz; i++) {
    unsigned row, col;
    double value;
    in_stream >> row >> col >> value;
    if (row >= rows || col >= cols) {
      std::cerr << "Error: Row or column index out of bounds in input."
                << std::endl;
      return 1;
    }
    triplets_A.emplace_back(row, col, value);
  }

  Eigen::SparseMatrix<double> matrix(rows, cols);
  matrix.setFromTriplets(triplets_A.begin(), triplets_A.end());

  // Read RHS vector
  Eigen::VectorXd rhs(rows);
  for (unsigned i = 0; i < rows; i++) {
    if (!(in_stream >> rhs[i])) {
      std::cerr << "Error: Insufficient elements in RHS vector." << std::endl;
      return 1;
    }
  }

  in_stream.close();

  // Create SparseLinearSystem
  SparseLinearSystem system{matrix, rhs};
  // write_sparsity_pattern(system.A, "sparsity_before.txt");

  // Reorder using Reverse Cuthill-McKee
  double starting_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  SparseLinearSystem reordered_system = reorder_system_rcm(system);
  double stopping_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  std::cout << "Time: " << stopping_time - starting_time << "ms" << std::endl;
  // write_sparsity_pattern(reordered_system.A, "sparsity_after.txt");
  /*
  std::cout << "Bandwidth:    " << compute_bandwidth(reordered_system.A)
            << std::endl;
  std::cout << "Original Size:    " << rows << std::endl;
  std::cout << "Percentage:    "
            << (1 - (double)compute_bandwidth(reordered_system.A) / rows) * 100
            << std::endl;
  */
  std::string output_filename = get_output_filename(input_filename);
  export_matrix(reordered_system.A, reordered_system.b, output_filename);
  return 0;
}
