#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace Eigen;
using namespace std;

vector<double> globalDegree;

int findIndex(vector<pair<int, double>> a, int x)
{
    for (int i = 0; i < a.size(); i++)
        if (a[i].first == x)
            return i;
    return -1;
}

bool compareDegree(int i, int j)
{
    return ::globalDegree[i] < ::globalDegree[j];
}

std::vector<int> reverse_cuthill_mckee(const Eigen::SparseMatrix<double> &A) {
    int n = A.rows();
    std::vector<int> perm(n, -1);
    std::vector<int> degree(n, 0);
    std::vector<bool> visited(n, false);

    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            degree[it.row()]++;
        }
    }

    std::queue<int> Q;
    int start = std::min_element(degree.begin(), degree.end()) - degree.begin();
    Q.push(start);
    visited[start] = true;
    int index = 0;

    while (!Q.empty()) {
        int v = Q.front();
        Q.pop();
        perm[index++] = v;

        std::vector<int> neighbors;
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, v); it; ++it) {
            if (!visited[it.col()]) {
                neighbors.push_back(it.col());
                visited[it.col()] = true;
            }
        }

        std::sort(neighbors.begin(), neighbors.end(), [&degree](int a, int b) {
            return degree[a] < degree[b];
        });

        for (int neighbor : neighbors) {
            Q.push(neighbor);
        }
    }

    std::reverse(perm.begin(), perm.end());
    return perm;
}

struct SparseLinearSystem {
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
};

SparseLinearSystem reorder_system_rcm(const SparseLinearSystem &system) {
    std::vector<int> perm = reverse_cuthill_mckee(system.A);
    Eigen::VectorXi perm_eigen = Eigen::Map<Eigen::VectorXi>(perm.data(), perm.size());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_matrix(perm_eigen);

    std::cout << "Permutation matrix size: " << perm_matrix.rows() << "x" << perm_matrix.cols() << std::endl;
    std::cout << "Original matrix size: " << system.A.rows() << "x" << system.A.cols() << std::endl;

    std::cout << "Permutation vector: ";
    for (int i = 0; i < perm.size(); ++i) {
        std::cout << perm[i] << " ";
    }
    std::cout << std::endl;

    // Apply permutation matrix
    Eigen::SparseMatrix<double> A_temp = perm_matrix.transpose() * system.A;
    std::cout << "Intermediate matrix size: " << A_temp.rows() << "x" << A_temp.cols() << std::endl;

    Eigen::SparseMatrix<double> A_reordered = A_temp * perm_matrix;
    Eigen::VectorXd b_reordered = perm_matrix.transpose() * system.b;

    SparseLinearSystem reordered_system;
    reordered_system.A = A_reordered;
    reordered_system.b = b_reordered;
    return reordered_system;
}

int main() {
    std::ifstream in_stream("problem1_complexity1.txt");
    if (!in_stream.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    unsigned nnz, rows, cols;
    in_stream >> nnz >> rows >> cols;

    assert(nnz <= rows * cols);

    std::vector<Eigen::Triplet<double>> triplets_A;
    triplets_A.reserve(nnz);

    // every next three entries correspond to values for a triplet
    for (unsigned i = 0; i < nnz; i++) {
        unsigned row, col;
        double value;
        in_stream >> row >> col >> value;
        if (row >= rows || col >= cols) {
            std::cerr << "Error: row or column index out of bounds" << std::endl;
            return 1;
        }
        triplets_A.emplace_back(row, col, value);
    }

    Eigen::SparseMatrix<double> matrix(rows, cols);
    matrix.setFromTriplets(triplets_A.begin(), triplets_A.end());

    // construct rhs vector
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rows);
    for (unsigned i = 0; i < rows; i++) {
        if (!(in_stream >> rhs[i])) {
            std::cerr << "Error: not enough elements in the input file for the RHS vector" << std::endl;
            return 1;
        }
    }

    // Create SparseLinearSystem
    SparseLinearSystem system;
    system.A = matrix;
    system.b = rhs;

    // Reorder the system using Reverse Cuthill-McKee
    SparseLinearSystem reordered_system = reorder_system_rcm(system);

    std::cout << "\nReordered Matrix:\n";
    for (int k = 0; k < reordered_system.A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(reordered_system.A, k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << std::endl;
        }
    }

    std::cout << "\nReordered RHS Vector:\n" << reordered_system.b.transpose() << std::endl;

    return 0;
}