#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/mesh_factory.h>
#include <lf/mesh/utils/utils.h>

#include <filesystem>
#include <iostream>

int main() {
  std::cout << "LehrFEM++ DEMO: mesh capabilities and functionality" << '\n';
  // Set complete file path to the sample mesh in Gmsh format
  const std::filesystem::path here = __FILE__;
  auto mesh_file = here.parent_path() / "../meshes/unit_square.msh";

  // Create a 2D mesh data structure from the information contained in
  // the file A factory object is in charge of
  // creating mesh entities and has to be initialized first.
  auto factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(factory), mesh_file.string());
  // Obtain pointer to read only mesh from the mesh reader object
  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh &mesh{*mesh_ptr};

  // Output general information on mesh; self-explanatory
  std::cout << "Mesh from file " << mesh_file.string() << ": ["
            << mesh.DimMesh() << ',' << mesh.DimWorld() << "] dim:" << '\n';
  std::cout << mesh.NumEntities(0) << " cells, " << mesh.NumEntities(1)
            << " edges, " << mesh.NumEntities(2) << " nodes" << '\n';
}
