#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh_factory.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>

#include <Eigen/SparseCore>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "linear_systems/sparse.hpp"

SparseLinearSystem generate_system(nlohmann::json config_data) {
  // set up rhs functions for problems we want to benchmark on
  // f for problem 1 of benchmarking PDEs (transferred to 2d)
  // see
  // https://www.sciencedirect.com/science/article/pii/S0167819109001252#sec5
  auto f1 = [](Eigen::Vector2d x) {
    return 2 * (x[0] - x[0] * x[0]) + 2 * (x[1] - x[1] * x[1]) + 1000;
  };
  std::vector<decltype(f1)> rhs_functions;
  rhs_functions.push_back(f1);

  unsigned selector_mesh = config_data["selector"];
  double scale = config_data["scale"];
  // generate mesh
  auto mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(selector_mesh, scale);
  // refine mesh
  const unsigned refsteps = config_data["refinement"];
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, refsteps);
  lf::refinement::MeshHierarchy &multi_mesh{*multi_mesh_p};
  const unsigned L = multi_mesh.NumLevels();
  mesh_p = multi_mesh.getMesh(L - 1);

  // Create HierarchicalFESpace
  const unsigned degree = config_data["degree"];
  const auto fe_space =
      std::make_shared<lf::fe::HierarchicScalarFESpace<double>>(mesh_p, degree);

  // define diffusion coefficient
  const lf::mesh::utils::MeshFunctionConstant mf_alpha(1);
  // define rhs load
  unsigned problem = config_data["problem"];
  lf::mesh::utils::MeshFunctionGlobal mf_load{rhs_functions.at(problem - 1)};

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

  return SparseLinearSystem(A_COO.makeSparse(), rhs);
}