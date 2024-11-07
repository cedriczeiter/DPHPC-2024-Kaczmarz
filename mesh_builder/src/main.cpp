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
#include "utils.hpp"


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
  generate_system(config_data).write_to_stream(std::cout);

  return 0;
}
