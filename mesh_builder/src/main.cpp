#include <Eigen/SparseCore>
#include <fstream>
#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh_factory.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>

#include <filesystem>
#include <iostream>

#include "utils.hpp"

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

  auto [A, rhs] = generate_system(config_data);

  // Save Matrix to use Kaczmarz solver
  return export_matrix(A, rhs, config_data["matrix_file"]);
}
