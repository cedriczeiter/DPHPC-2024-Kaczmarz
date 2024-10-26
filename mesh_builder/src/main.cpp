#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh_factory.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>

#include <Eigen/SparseCore>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "linear_systems/sparse.hpp"
#include <nlohmann/json.hpp>

SparseLinearSystem generate_system(
    nlohmann::json config_data) {
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

  return SparseLinearSystem(A_COO.makeSparse(), rhs);
}

int main(int argc, char **argv) {
  // Set complete file path to the configuration file;
  const std::filesystem::path here = __FILE__;
  auto config_file_path = here.parent_path() / "../configuration.json";

  // read in data from configuration file; right now we dont have many params in
  // there yet, but this mightchange, so having everything in a config file will
  // be much easier
  std::ifstream config_file(config_file_path.string());
  if (!config_file.is_open()) {
    std::cerr << "Error opening config file!" << std::endl;
    return 1;
  }
  nlohmann::json config_data;
  config_file >> config_data;
  config_file.close();

  // Save Matrix to use Kaczmarz solver
  generate_system(config_data).write_to_file(config_data["matrix_file"]);

  return 0;
}
