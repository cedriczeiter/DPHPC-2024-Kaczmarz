#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;

#include <Eigen/Sparse>
#include <algorithm>
#include <numeric>
#include <queue>
#include <vector>

std::vector<int> reverse_cuthill_mckee(const Eigen::SparseMatrix<double> &A) {
  int n = A.rows();
  std::vector<int> perm(n, -1);         // Output permutation
  std::vector<int> degree(n, 0);        // Degree of each node
  std::vector<bool> visited(n, false);  // Visited nodes

  // Compute degree of each node
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      if (it.row() != it.col()) {  // Ignore self-loops
        degree[it.row()]++;
        degree[it.col()]++;
      }
    }
  }

  // Result permutation index
  int index = 0;

  // Perform BFS for all connected components
  for (int i = 0; i < n; ++i) {
    if (!visited[i]) {
      // Start BFS from the node with the minimum degree in this component
      std::queue<int> Q;
      Q.push(i);
      visited[i] = true;

      while (!Q.empty()) {
        int v = Q.front();
        Q.pop();
        perm[index++] = v;

        // Collect unvisited neighbors
        std::vector<int> neighbors;
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, v); it; ++it) {
          int neighbor = it.col();
          if (!visited[neighbor]) {
            neighbors.push_back(neighbor);
            visited[neighbor] = true;
          }
        }

        // Sort neighbors by degree
        std::sort(neighbors.begin(), neighbors.end(),
                  [&degree](int a, int b) { return degree[a] < degree[b]; });

        // Add sorted neighbors to the queue
        for (int neighbor : neighbors) {
          Q.push(neighbor);
        }
      }
    }
  }

  // Reverse the permutation for RCM order
  std::reverse(perm.begin(), perm.end());
  return perm;
}

struct SparseLinearSystem {
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd b;
};

SparseLinearSystem reorder_system_rcm(const SparseLinearSystem &system) {
  std::vector<int> perm = reverse_cuthill_mckee(system.A);
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
  std::ifstream in_stream("problem1_complexity2.txt");
  if (!in_stream.is_open()) {
    std::cerr << "Error: Could not open input file." << std::endl;
    return 1;
  }

  unsigned nnz, rows, cols;
  in_stream >> nnz >> rows >> cols;

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

  // Create SparseLinearSystem
  SparseLinearSystem system{matrix, rhs};

  // Reorder using Reverse Cuthill-McKee
  SparseLinearSystem reordered_system = reorder_system_rcm(system);

  std::cout << "\nReordered Matrix:\n";
  for (int k = 0; k < reordered_system.A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(reordered_system.A, k);
         it; ++it) {
      std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value()
                << std::endl;
    }
  }

  std::cout << "\nReordered RHS Vector:\n"
            << reordered_system.b.transpose() << std::endl;

  return 0;
}
