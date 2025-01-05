#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh_factory.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/SparseCore>
#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>

#include "linear_systems/sparse.hpp"

struct PositionHint {
  double x, y;
};

struct Discretization {
  SparseLinearSystem sys;
  std::vector<PositionHint> position_hints;
};

Discretization generate_discretization(nlohmann::json config_data) {
  // set up rhs functions for problems we want to benchmark on
  // f for problem 1 of benchmarking PDEs (transferred to 2d)
  // we craft analytical solutions for: u = xy(1-x)(1-y)
  // problem one: -lapl(u) + 1000 = F
  auto f1 = [](Eigen::Vector2d x) {
    return 2 * (x[0] - x[0] * x[0]) + 2 * (x[1] - x[1] * x[1]) + 1000;
  };
  auto alpha1 = [](Eigen::Vector2d) { return 1.; };
  auto gamma1 = [](Eigen::Vector2d) { return 1; };

  // problem 2: -lapl(u) + (x*x + y*y)u = F
  auto f2 = [](Eigen::Vector2d x_vec) {
    const double x = x_vec[0];
    const double y = x_vec[1];
    return 2 * (x - x * x) + 2 * (y - y * y) +
           (x * x + y * y) * (x - x * x) * (y - y * y);
  };
  auto alpha2 = [](Eigen::Vector2d) { return 1.; };
  auto gamma2 = [](Eigen::Vector2d x_vec) {
    const double x = x_vec[0];
    const double y = x_vec[1];
    return (x * x + y * y);
  };

  // problem 3: -div(x*y*grad(u)) + x*y*u = F
  auto f3 = [](Eigen::Vector2d x_vec) {
    const double x = x_vec[0];
    const double y = x_vec[1];
    return -((1 - 4 * x) * (y * y - y * y * y) +
             (1 - 4 * y) * (x * x - x * x * x)) +
           (x * x - x * x * x) * (y * y - y * y * y);
  };
  auto alpha3 = [](Eigen::Vector2d x) { return x[0] * x[1]; };
  auto gamma3 = [](Eigen::Vector2d x) { return x[0] * x[1]; };

  std::vector<std::function<double(Eigen::Vector2d)>> rhs_functions;
  rhs_functions.push_back(f1);
  rhs_functions.push_back(f2);
  rhs_functions.push_back(f3);

  std::vector<std::function<double(Eigen::Vector2d)>> alpha_functions;
  alpha_functions.push_back(alpha1);
  alpha_functions.push_back(alpha2);
  alpha_functions.push_back(alpha3);

  std::vector<std::function<double(Eigen::Vector2d)>> gamma_functions;
  gamma_functions.push_back(gamma1);
  gamma_functions.push_back(gamma2);
  gamma_functions.push_back(gamma3);

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
  const unsigned degree =
      config_data["degree"];  // two degrees to choose from at the moment
  std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<double>> fe_space;

  if (degree == 1) {
    fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
  } else if (degree == 2) {
    fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
  } else {
    fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO3<double>>(mesh_p);
  }

  const lf::assemble::DofHandler& dofh = fe_space->LocGlobMap();

  std::vector<PositionHint> position_hints;

  // For each global dof, get the associated entity:
  for (int dof_idx = 0; dof_idx < dofh.NumDofs(); ++dof_idx) {
    const lf::mesh::Entity& dof_entity = dofh.Entity(dof_idx);

    // Get the geometry for this entity
    auto geom = dof_entity.Geometry();

    // Now retrieve the coordinates of its corners:
    //   - For a vertex, it's a single point.
    //   - For an edge, it's the two endpoints.
    //   - For a cell, it's all corner points, etc.
    Eigen::MatrixXd corners = geom->Global(dof_entity.RefEl().NodeCoords());

    // The points should be in 2D
    assert(corners.rows() == 2);

    // Average the points into a single point
    Eigen::VectorXd point = corners.rowwise().mean();

    position_hints.emplace_back(point[0], point[1]);
  }

  unsigned problem = config_data["problem"];

  // define diffusion coefficient
  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha_functions.at(problem - 1)};

  // define rhs load
  lf::mesh::utils::MeshFunctionGlobal mf_load{rhs_functions.at(problem - 1)};

  // define reaction coefficient
  lf::mesh::utils::MeshFunctionGlobal mf_gamma{gamma_functions.at(problem - 1)};

  // Assemble the system matrix and right hand side
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(fe_space->LocGlobMap().NumDofs());
  //const lf::assemble::DofHandler &dofh = fe_space->LocGlobMap();
  lf::assemble::COOMatrix<double> A_COO(dofh.NumDofs(), dofh.NumDofs());

  lf::uscalfe::ReactionDiffusionElementMatrixProvider<
      double, decltype(mf_alpha), decltype(mf_gamma)>
      element_matrix_provider(fe_space, mf_alpha, mf_gamma);
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

  return { SparseLinearSystem(A_COO.makeSparse(), rhs), position_hints };
}

int main(int argc, char **argv) {
  nlohmann::json config_data;

  // read in data from configuration file (or stdin); right now we dont have
  // many params in there yet, but this might change, so having everything in a
  // config file will be much easier
  if (argc == 2) {
    // there is a command-line argument -> interpret it as path to config file

    std::ifstream config_file(argv[1]);
    if (!config_file.is_open()) {
      std::cerr << "Can't open config file " << argv[1] << std::endl;
      return 1;
    }
    config_file >> config_data;
  } else {
    // there are no or too many command-line arguments -> read config from stdin

    std::cin >> config_data;
  }

  // Output the LSE to use Kaczmarz solver
  generate_discretization(config_data).sys.write_to_stream(std::cout);

  // TODO: output the position hints

  return 0;
}
