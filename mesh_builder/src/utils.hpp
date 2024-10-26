#ifndef UTILS_HPP
#define UTILS_HPP

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

// idea for export Linear System:
// 1st entry: nnz in Matrix, 2nd and 3rd entry: rows/cols of Matrix, then
// triplets printed out, then values of RHS Vector
int export_matrix(Eigen::SparseMatrix<double> A, Eigen::VectorXd rhs,
                  std::string path) {
  const unsigned rows = A.rows();
  const unsigned cols = A.cols();
  std::ofstream outFile(path);
  if (!outFile.is_open()) {
    std::cerr << "Error opening file for writing!" << std::endl;
    return 1;
  }
  outFile << A.nonZeros() << std::endl;
  outFile << A.rows() << " " << A.cols() << std::endl;

  // print values of matrix
  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      outFile << it.row() << " " << it.col() << " " << it.value() << std::endl;
    }
  }
  // print values of vector
  for (int i = 0; i < rhs.size(); i++)
    outFile << rhs[i] << std::endl;

  return 0;
}

std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd>
generate_system(nlohmann::json config_data) {
  unsigned selector_mesh = config_data["selector"];
  double scale = config_data["scale"];
  // generate mesh
  auto mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(selector_mesh, scale);

  // Create HierarchicalFESpace
  const unsigned degree = config_data["degree"];
  const auto fe_space =
      std::make_shared<lf::fe::HierarchicScalarFESpace<double>>(mesh_p, degree);

  // define diffusion coefficient
  const lf::mesh::utils::MeshFunctionConstant mf_alpha(1);
  // define rhs load
  auto mf_load = lf::mesh::utils::MeshFunctionConstant(1.);

  // Assemble the system matrix and right hand side
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(fe_space->LocGlobMap().NumDofs());
  const lf::assemble::DofHandler &dofh = fe_space->LocGlobMap();
  lf::assemble::COOMatrix<double> A_COO(dofh.NumDofs(), dofh.NumDofs());

  lf::fe::DiffusionElementMatrixProvider element_matrix_provider(fe_space,
                                                                 mf_alpha);
  AssembleMatrixLocally(0, dofh, dofh, element_matrix_provider, A_COO);
  lf::fe::ScalarLoadElementVectorProvider element_vector_provider(fe_space,
                                                                  mf_load);
  AssembleVectorLocally(0, dofh, element_vector_provider, rhs);

  // Enforce zero dirichlet boundary conditions

  const auto boundary = lf::mesh::utils::flagEntitiesOnBoundary(mesh_p);
  const auto selector = [&](unsigned int idx) -> std::pair<bool, double> {
    const auto &entity = dofh.Entity(idx);
    return {entity.Codim() > 0 && boundary(entity), 0};
  };
  FixFlaggedSolutionComponents(selector, A_COO, rhs);

  return std::make_pair(A_COO.makeSparse(), rhs);
}
#endif
