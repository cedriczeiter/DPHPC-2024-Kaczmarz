#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>

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

class ReorderingSSM {
private:
    vector<vector<double>> _matrix;

public:
    // Constructor and Destructor
    ReorderingSSM(vector<vector<double>> m)
    {
        _matrix = m;
    }

    ReorderingSSM() {}
    ~ReorderingSSM() {}

    // Method to generate degree of all the nodes
    vector<double> degreeGenerator()
    {
        vector<double> degrees;
        for (int i = 0; i < _matrix.size(); i++) {
            double count = 0;
            for (int j = 0; j < _matrix[0].size(); j++) {
                count += _matrix[i][j];
            }
            degrees.push_back(count);
        }
        return degrees;
    }

    // Cuthill-McKee algorithm implementation
    vector<int> CuthillMckee()
    {
        vector<double> degrees = degreeGenerator();
        ::globalDegree = degrees;

        queue<int> Q;
        vector<int> R;
        vector<pair<int, double>> notVisited;

        for (int i = 0; i < degrees.size(); i++)
            notVisited.push_back(make_pair(i, degrees[i]));

        // BFS even for disconnected components
        while (notVisited.size()) {
            int minNodeIndex = 0;
            for (int i = 0; i < notVisited.size(); i++)
                if (notVisited[i].second < notVisited[minNodeIndex].second)
                    minNodeIndex = i;

            Q.push(notVisited[minNodeIndex].first);
            notVisited.erase(notVisited.begin() + findIndex(notVisited, notVisited[Q.front()].first));

            // Simple BFS
            while (!Q.empty()) {
                vector<int> toSort;
                for (int i = 0; i < _matrix[0].size(); i++) {
                    if (i != Q.front() && _matrix[Q.front()][i] == 1 && findIndex(notVisited, i) != -1) {
                        toSort.push_back(i);
                        notVisited.erase(notVisited.begin() + findIndex(notVisited, i));
                    }
                }

                sort(toSort.begin(), toSort.end(), compareDegree);
                for (int i = 0; i < toSort.size(); i++)
                    Q.push(toSort[i]);

                R.push_back(Q.front());
                Q.pop();
            }
        }

        return R;
    }

    // Reverse Cuthill-McKee algorithm implementation
    vector<int> ReverseCuthillMckee()
    {
        vector<int> cuthill = CuthillMckee();

        int n = cuthill.size();
        if (n % 2 == 0)
            n -= 1;

        n = n / 2;
        for (int i = 0; i <= n; i++) {
            int j = cuthill[cuthill.size() - 1 - i];
            cuthill[cuthill.size() - 1 - i] = cuthill[i];
            cuthill[i] = j;
        }

        return cuthill;
    }
};

void printSparseMatrix(const SparseMatrix<int>& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int value = matrix.coeff(i, j); // coeff() returns 0 if no entry exists
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}

SparseMatrix<int> reorderMatrix(const SparseMatrix<int>& matrix, const std::vector<int>& order) {
    int n = matrix.rows();
    SparseMatrix<int> reorderedMatrix(n, n);
    std::vector<int> newIndex(n);
    for (int i = 0; i < n; ++i) {
        newIndex[order[i]] = i;
    }

    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (SparseMatrix<int>::InnerIterator it(matrix, k); it; ++it) {
            int new_row = newIndex[it.row()];
            int new_col = newIndex[it.col()];
            reorderedMatrix.insert(new_row, new_col) = it.value();
        }
    }

    reorderedMatrix.makeCompressed();
    return reorderedMatrix;
}

int main() {
    // Define a sparse matrix (example)
    SparseMatrix<int> matrix(10, 10);
    std::vector<Eigen::Triplet<int>> triplets;
    triplets.push_back(Eigen::Triplet<int>(0, 1, 1));
    triplets.push_back(Eigen::Triplet<int>(0, 6, 1));
    triplets.push_back(Eigen::Triplet<int>(0, 8, 1));
    triplets.push_back(Eigen::Triplet<int>(1, 0, 1));
    triplets.push_back(Eigen::Triplet<int>(1, 4, 1));
    triplets.push_back(Eigen::Triplet<int>(1, 6, 1));
    triplets.push_back(Eigen::Triplet<int>(1, 9, 1));
    triplets.push_back(Eigen::Triplet<int>(2, 4, 1));
    triplets.push_back(Eigen::Triplet<int>(2, 6, 1));
    triplets.push_back(Eigen::Triplet<int>(3, 4, 1));
    triplets.push_back(Eigen::Triplet<int>(3, 5, 1));
    triplets.push_back(Eigen::Triplet<int>(3, 8, 1));
    triplets.push_back(Eigen::Triplet<int>(4, 1, 1));
    triplets.push_back(Eigen::Triplet<int>(4, 2, 1));
    triplets.push_back(Eigen::Triplet<int>(4, 3, 1));
    triplets.push_back(Eigen::Triplet<int>(4, 5, 1));
    triplets.push_back(Eigen::Triplet<int>(4, 9, 1));
    triplets.push_back(Eigen::Triplet<int>(5, 3, 1));
    triplets.push_back(Eigen::Triplet<int>(5, 4, 1));
    triplets.push_back(Eigen::Triplet<int>(6, 0, 1));
    triplets.push_back(Eigen::Triplet<int>(6, 1, 1));
    triplets.push_back(Eigen::Triplet<int>(6, 2, 1));
    triplets.push_back(Eigen::Triplet<int>(7, 8, 1));
    triplets.push_back(Eigen::Triplet<int>(7, 9, 1));
    triplets.push_back(Eigen::Triplet<int>(8, 0, 1));
    triplets.push_back(Eigen::Triplet<int>(8, 3, 1));
    triplets.push_back(Eigen::Triplet<int>(8, 7, 1));
    triplets.push_back(Eigen::Triplet<int>(9, 1, 1));
    triplets.push_back(Eigen::Triplet<int>(9, 4, 1));
    triplets.push_back(Eigen::Triplet<int>(9, 7, 1));
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "Original Matrix:\n";
    printSparseMatrix(matrix);

    // Convert sparse matrix to dense matrix format for RCM processing
    int n = matrix.rows();
    vector<vector<double>> denseMatrix(n, vector<double>(n, 0));
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (SparseMatrix<int>::InnerIterator it(matrix, k); it; ++it) {
            denseMatrix[it.row()][it.col()] = it.value();
        }
    }

    // Use the second RCM implementation
    ReorderingSSM mtxReorder(denseMatrix);
    vector<int> rcmOrder = mtxReorder.ReverseCuthillMckee();

    std::cout << "Reordered Matrix:\n";
    SparseMatrix<int> reorderedMatrix = reorderMatrix(matrix, rcmOrder);
    printSparseMatrix(reorderedMatrix);

    return 0;
}


// RCM Code taken from:
// https://www.geeksforgeeks.org/reverse-cuthill-mckee-algorithm/