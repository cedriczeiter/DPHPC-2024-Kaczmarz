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

void reorder_system_rcm(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, Eigen::SparseMatrix<double> &A_reordered, Eigen::VectorXd &b_reordered) {
    std::vector<int> perm = reverse_cuthill_mckee(A);
    Eigen::VectorXi perm_eigen = Eigen::Map<Eigen::VectorXi>(perm.data(), perm.size());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_matrix(perm_eigen);
    A_reordered = perm_matrix.transpose() * A * perm_matrix;
    b_reordered = perm_matrix.transpose() * b;
}

void printSparseMatrix(const Eigen::SparseMatrix<double> &matrix) {
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << endl;
        }
    }
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
        triplets_A.emplace_back(row, col, value);
    }

    Eigen::SparseMatrix<double> matrix(rows, cols);
    matrix.setFromTriplets(triplets_A.begin(), triplets_A.end());

    // construct rhs vector
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rows);
    for (unsigned i = 0; i < rows; i++) {
        in_stream >> rhs[i];
    }

    // Close the input file
    in_stream.close();

    // Reorder the system using Reverse Cuthill-McKee
    Eigen::SparseMatrix<double> A_reordered;
    Eigen::VectorXd b_reordered;
    reorder_system_rcm(matrix, rhs, A_reordered, b_reordered);

    std::cout << "\nReordered Matrix:\n";
    printSparseMatrix(A_reordered);

    std::cout << "\nReordered RHS Vector:\n" << b_reordered.transpose() << std::endl;

    return 0;
}




// RCM Code taken from:
// https://www.geeksforgeeks.org/reverse-cuthill-mckee-algorithm/